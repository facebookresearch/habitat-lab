#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import time

import faster_fifo
import numpy as np
import torch
import tqdm
from gym import spaces
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from habitat import logger
from habitat.utils import profiling_wrapper
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    add_signal_handlers,
    init_distrib_slurm,
    is_slurm_batch_job,
    load_resume_state,
    rank0_only,
    requeue_job,
    save_resume_state,
)
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_baselines.rl.vsf.actor_worker import (
    construct_actor_workers,
    int_action_plugin,
)
from habitat_baselines.rl.vsf.policy_worker import (
    PolicyWorker,
    PolicyWorkerProcess,
)
from habitat_baselines.rl.vsf.report_worker import ReportWorker
from habitat_baselines.rl.vsf.task_enums import ReportWorkerTasks
from habitat_baselines.rl.vsf.timing import Timing
from habitat_baselines.rl.vsf.vsf_rollout_storage import VSFRolloutStorage
from habitat_baselines.utils.common import action_to_velocity_control


@baseline_registry.register_trainer(name="vsf")
class VSFTrainer(PPOTrainer):
    def _init_train(self):
        if self.config.RL.DDPPO.force_distributed:
            self._is_distributed = True

        if is_slurm_batch_job():
            add_signal_handlers()

        if self._is_distributed:
            local_rank, tcp_store = init_distrib_slurm(
                self.config.RL.DDPPO.distrib_backend
            )
            if rank0_only():
                logger.info(
                    "Initialized DD-PPO with {} workers".format(
                        torch.distributed.get_world_size()
                    )
                )

            self.config.defrost()
            self.config.TORCH_GPU_ID = local_rank
            self.config.SIMULATOR_GPU_ID = local_rank
            # Multiply by the number of simulators to make sure they also get unique seeds
            self.config.TASK_CONFIG.SEED += (
                torch.distributed.get_rank() * self.config.NUM_ENVIRONMENTS
            )
            self.config.freeze()

            random.seed(self.config.TASK_CONFIG.SEED)
            np.random.seed(self.config.TASK_CONFIG.SEED)
            torch.manual_seed(self.config.TASK_CONFIG.SEED)
            self.num_rollouts_done_store = torch.distributed.PrefixStore(
                "rollout_tracker", tcp_store
            )
            self.num_rollouts_done_store.set("num_done", "0")

        if rank0_only() and self.config.VERBOSE:
            logger.info(f"config: {self.config}")

        profiling_wrapper.configure(
            capture_start_step=self.config.PROFILING.CAPTURE_START_STEP,
            num_steps_to_capture=self.config.PROFILING.NUM_STEPS_TO_CAPTURE,
        )

        self.mp_ctx = torch.multiprocessing.get_context("forkserver")
        self.report_queue = faster_fifo.Queue(
            max_size_bytes=8 * 1024 * 1024 * self.config.NUM_ENVIRONMENTS
        )
        self.policy_worker_queue = faster_fifo.Queue(
            max_size_bytes=1 * 1024 * self.config.NUM_ENVIRONMENTS
        )

        self.actor_workers, self.actor_worker_queues = construct_actor_workers(
            self.config,
            get_env_class(self.config.ENV_NAME),
            self.mp_ctx,
            self.policy_worker_queue,
            self.report_queue,
        )

        init_reports = []
        while len(init_reports) < len(self.actor_workers):
            init_reports += self.report_queue.get_many()

        self.report_worker = ReportWorker(
            self.mp_ctx, self.config, self.report_queue
        )

        action_space = init_reports[0]["act_space"]

        if self.using_velocity_ctrl:
            self.policy_action_space = action_space["VELOCITY_CONTROL"]
            action_shape = (2,)
            discrete_actions = False
        else:
            self.policy_action_space = action_space
            action_shape = None
            discrete_actions = True

        ppo_cfg = self.config.RL.PPO
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.config.TORCH_GPU_ID)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if rank0_only() and not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self.obs_space = init_reports[0]["obs_space"]
        self._setup_actor_critic_agent(ppo_cfg)
        if self._is_distributed:
            self.agent.init_distributed(find_unused_params=True)

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        obs_space = self.obs_space
        if self._static_encoder:
            self._encoder = self.actor_critic.net.visual_encoder
            obs_space = spaces.Dict(
                {
                    "visual_features": spaces.Box(
                        low=np.finfo(np.float32).min,
                        high=np.finfo(np.float32).max,
                        shape=self._encoder.output_shape,
                        dtype=np.float32,
                    ),
                    **obs_space.spaces,
                }
            )

        n_policy_workers = 1
        self.rollouts = VSFRolloutStorage(
            self.mp_ctx,
            n_policy_workers,
            ppo_cfg.num_steps,
            len(self.actor_workers),
            obs_space,
            self.policy_action_space,
            ppo_cfg.hidden_size,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
            action_shape=action_shape,
            discrete_actions=discrete_actions,
        )

        transfer_buffers = self.rollouts.buffers[0]
        (
            transfer_buffers.map_in_place(lambda t: t.clone().contiguous())
            .map_in_place(lambda t: t.share_memory_())
            .map_in_place(lambda t: t.numpy())
        )

        [w.set_transfer_buffers(transfer_buffers) for w in self.actor_workers]
        [w.reset() for w in self.actor_workers]

        self.rollouts.to(self.device)
        self.rollouts.share_memory_()
        self.actor_critic = self.actor_critic.share_memory()

        policy_worker_args = (
            self.policy_worker_queue,
            self.actor_worker_queues,
            self.report_queue,
            self.rollouts,
            transfer_buffers,
            self.actor_critic,
        )

        self.policy_workers = [
            PolicyWorker(self.mp_ctx, i, *policy_worker_args)
            for i in range(1, n_policy_workers)
        ]
        self._policy_worker_impl = PolicyWorkerProcess(
            0,
            *policy_worker_args,
            self.device,
        )

        self.timer = Timing()

    @profiling_wrapper.RangeContext("_update_agent")
    def _update_agent(self):
        ppo_cfg = self.config.RL.PPO
        t_update_model = time.time()

        with self.timer.avg_time("compute returns"):
            self.rollouts.compute_returns(
                ppo_cfg.use_gae,
                ppo_cfg.gamma,
                ppo_cfg.tau,
                False,
                self.actor_critic,
            )

        with self.timer.avg_time("update agent"):
            self.agent.train()

            value_loss, action_loss, dist_entropy = self.agent.update(
                self.rollouts
            )

        with self.timer.avg_time("after update"):
            self.rollouts.after_update()

        self.pth_time += time.time() - t_update_model

        return (
            value_loss,
            action_loss,
            dist_entropy,
        )

    def should_end_early(self, rollout_step) -> bool:
        if not self._is_distributed:
            return False
        # This is where the preemption of workers happens.  If a
        # worker detects it will be a straggler, it preempts itself!
        return (
            rollout_step
            >= self.config.RL.PPO.num_steps * self.SHORT_ROLLOUT_THRESHOLD
        ) and int(self.num_rollouts_done_store.get("num_done")) >= (
            self.config.RL.DDPPO.sync_frac * torch.distributed.get_world_size()
        )

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training DD/PPO.

        Returns:
            None
        """

        self._init_train()

        count_checkpoints = 0
        prev_time = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: 1 - self.percent_done(),
        )

        resume_state = load_resume_state(self.config)
        if resume_state is not None:
            self.agent.load_state_dict(resume_state["state_dict"])
            self.agent.optimizer.load_state_dict(resume_state["optim_state"])
            lr_scheduler.load_state_dict(resume_state["lr_sched_state"])

            requeue_stats = resume_state["requeue_stats"]
            self.env_time = requeue_stats["env_time"]
            self.pth_time = requeue_stats["pth_time"]
            self.num_steps_done = requeue_stats["num_steps_done"]
            self.num_updates_done = requeue_stats["num_updates_done"]
            self._last_checkpoint_percent = requeue_stats[
                "_last_checkpoint_percent"
            ]
            count_checkpoints = requeue_stats["count_checkpoints"]
            prev_time = requeue_stats["prev_time"]

            self.running_episode_stats = requeue_stats["running_episode_stats"]
            self.window_episode_stats.update(
                requeue_stats["window_episode_stats"]
            )

        ppo_cfg = self.config.RL.PPO

        while not self.is_done():
            profiling_wrapper.on_start_step()

            if ppo_cfg.use_linear_clip_decay:
                self.agent.clip_param = ppo_cfg.clip_param * (
                    1 - self.percent_done()
                )

            if False and rank0_only() and self._should_save_resume_state():
                requeue_stats = dict(
                    env_time=self.env_time,
                    pth_time=self.pth_time,
                    count_checkpoints=count_checkpoints,
                    num_steps_done=self.num_steps_done,
                    num_updates_done=self.num_updates_done,
                    _last_checkpoint_percent=self._last_checkpoint_percent,
                    prev_time=(time.time() - self.t_start) + prev_time,
                    running_episode_stats=self.running_episode_stats,
                    window_episode_stats=dict(self.window_episode_stats),
                )

                save_resume_state(
                    dict(
                        state_dict=self.agent.state_dict(),
                        optim_state=self.agent.optimizer.state_dict(),
                        lr_sched_state=lr_scheduler.state_dict(),
                        config=self.config,
                        requeue_stats=requeue_stats,
                    ),
                    self.config,
                )

            if EXIT.is_set():
                profiling_wrapper.range_pop()  # train update

                requeue_job()
                break

            self.agent.eval()

            with torch.no_grad():
                self._policy_worker_impl.run_one_epoch()
                self.rollouts.all_done.wait()

            (
                value_loss,
                action_loss,
                dist_entropy,
            ) = self._update_agent()

            self.report_queue.put_many(
                [
                    (
                        ReportWorkerTasks.learner_update,
                        dict(
                            value_loss=float(value_loss),
                            action_loss=float(action_loss),
                            dist_entropy=float(dist_entropy),
                        ),
                    ),
                    (ReportWorkerTasks.learner_timing, self.timer),
                ]
            )

            if ppo_cfg.use_linear_lr_decay:
                lr_scheduler.step()  # type: ignore

            self.num_updates_done += 1

        [
            w.close()
            for w in self.actor_workers
            + self.policy_workers
            + [self.report_worker]
        ]
        [
            w.join()
            for w in self.actor_workers
            + self.policy_workers
            + [self.report_worker]
        ]
