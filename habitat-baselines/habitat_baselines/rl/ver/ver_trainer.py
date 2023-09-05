#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
import random
import time
from typing import List

import numpy as np
import torch

from habitat import logger
from habitat.config import read_write
from habitat.utils import profiling_wrapper
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_spec import EnvironmentSpec
from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    add_signal_handlers,
    get_distrib_size,
    get_free_port_distributed,
    get_main_addr,
    init_distrib_slurm,
    is_slurm_batch_job,
    load_resume_state,
    rank0_only,
    requeue_job,
    save_resume_state,
)
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet  # noqa: F401.
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_baselines.rl.ppo.single_agent_access_mgr import (
    get_rollout_obs_space,
)
from habitat_baselines.rl.ver.environment_worker import (
    build_action_plugin_from_policy_action_space,
    construct_environment_workers,
)
from habitat_baselines.rl.ver.inference_worker import (
    InferenceWorker,
    InferenceWorkerProcess,
)
from habitat_baselines.rl.ver.preemption_decider import PreemptionDeciderWorker
from habitat_baselines.rl.ver.report_worker import ReportWorker
from habitat_baselines.rl.ver.task_enums import ReportWorkerTasks
from habitat_baselines.rl.ver.ver_rollout_storage import VERRolloutStorage
from habitat_baselines.rl.ver.worker_common import (
    InferenceWorkerSync,
    WorkerBase,
    WorkerQueues,
)
from habitat_baselines.utils.common import cosine_decay, inference_mode
from habitat_baselines.utils.timing import Timing

try:
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
except AttributeError:
    pass


@baseline_registry.register_trainer(name="ver")
class VERTrainer(PPOTrainer):
    def _create_agent(self, resume_state, **kwargs):
        self._create_obs_transforms()
        return baseline_registry.get_agent_access_mgr(
            self.config.habitat_baselines.rl.agent.type
        )(
            config=self.config,
            env_spec=self._env_spec,
            is_distrib=self._is_distributed,
            device=self.device,
            resume_state=resume_state,
            num_envs=self.config.habitat_baselines.num_environments,
            percent_done_fn=self.percent_done,
            **kwargs,
        )

    def _init_train(self, resume_state):
        assert (
            not self.config.habitat.simulator.renderer.enable_batch_renderer
        ), "VER trainer does not support batch rendering."

        if self._is_distributed:
            local_rank, world_rank, _ = get_distrib_size()

            with read_write(self.config):
                self.config.habitat_baselines.torch_gpu_id = local_rank
                self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = (
                    local_rank
                )
                # Multiply by the number of simulators to make sure they also get unique seeds
                self.config.habitat.seed += (
                    world_rank * self.config.habitat_baselines.num_environments
                )

        random.seed(self.config.habitat.seed)
        np.random.seed(self.config.habitat.seed)
        torch.manual_seed(self.config.habitat.seed)

        self.mp_ctx = torch.multiprocessing.get_context("forkserver")
        self.queues = WorkerQueues(
            self.config.habitat_baselines.num_environments
        )
        self.environment_workers = construct_environment_workers(
            self.config,
            self.mp_ctx,
            self.queues,
        )
        [ew.start() for ew in self.environment_workers]
        [ew.reset() for ew in self.environment_workers]

        if self.config.habitat_baselines.rl.ddppo.force_distributed:
            self._is_distributed = True

        if is_slurm_batch_job():
            add_signal_handlers()

        if self._is_distributed:
            local_rank, tcp_store = init_distrib_slurm(
                self.config.habitat_baselines.rl.ddppo.distrib_backend
            )
            if rank0_only():
                logger.info(
                    "Initialized VER+DD-PPO with {} workers".format(
                        torch.distributed.get_world_size()
                    )
                )
        else:
            logger.info("Initialized VER")
            tcp_store = None

        self._last_should_end_val = None
        self._last_should_end_calc_time = 0

        if rank0_only() and self.config.habitat_baselines.verbose:
            logger.info(f"config: {self.config}")

        profiling_wrapper.configure(
            capture_start_step=self.config.habitat_baselines.profiling.capture_start_step,
            num_steps_to_capture=self.config.habitat_baselines.profiling.num_steps_to_capture,
        )

        self._all_workers: List[WorkerBase] = []

        if self._is_distributed:
            world_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            world_rank = 0
            world_size = 1

        self._my_t_zero = time.perf_counter()
        self.preemption_decider = PreemptionDeciderWorker(
            self.mp_ctx,
            get_main_addr(),
            get_free_port_distributed("preemption", tcp_store),
            world_rank,
            world_size,
            self.config,
            self.queues,
            self._my_t_zero,
        )

        has_report_resume_state = (
            resume_state is not None
            and "report_worker_state" in resume_state["requeue_stats"]
        )
        run_id = None
        if (
            has_report_resume_state
            and resume_state["requeue_stats"]["report_worker_state"]
            is not None
        ):
            run_id = resume_state["requeue_stats"]["report_worker_state"][
                "run_id"
            ]

        self.report_worker = ReportWorker(
            self.mp_ctx,
            get_free_port_distributed("report", tcp_store),
            self.config,
            self.queues.report,
            self._my_t_zero,
            self.num_steps_done,
            run_id=run_id,
        )

        if has_report_resume_state:
            self.report_worker.load_state_dict(
                resume_state["requeue_stats"]["report_worker_state"]
            )

        init_reports = [self.environment_workers[0].get_init_report()]

        action_space = init_reports[0]["act_space"]

        [
            ew.set_action_plugin(
                build_action_plugin_from_policy_action_space(action_space)
            )
            for ew in self.environment_workers
        ]

        ppo_cfg = self.config.habitat_baselines.rl.ppo
        if torch.cuda.is_available():
            self.device = torch.device(
                "cuda", self.config.habitat_baselines.torch_gpu_id
            )
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if rank0_only() and not os.path.exists(
            self.config.habitat_baselines.checkpoint_folder
        ):
            os.makedirs(self.config.habitat_baselines.checkpoint_folder)

        actor_obs_space = init_reports[0]["obs_space"]
        self._env_spec = EnvironmentSpec(
            observation_space=copy.deepcopy(actor_obs_space),
            action_space=action_space,
            orig_action_space=action_space,
        )
        self.ver_config = self.config.habitat_baselines.rl.ver

        self._agent = self._create_agent(
            resume_state, lr_schedule_fn=cosine_decay
        )

        rollouts_obs_space = get_rollout_obs_space(
            copy.deepcopy(self._env_spec.observation_space),
            self._agent.actor_critic,
            self.config,
        )
        storage_kwargs = {
            "variable_experience": self.ver_config.variable_experience,
            "numsteps": ppo_cfg.num_steps,
            "num_envs": len(self.environment_workers),
            "action_space": self._env_spec.action_space,
            "actor_critic": self._agent.actor_critic,
            "observation_space": rollouts_obs_space,
        }

        def create_ver_rollouts_fn(
            device,
            **kwargs,
        ):
            with inference_mode():
                rollouts = VERRolloutStorage(**storage_kwargs)
                rollouts.to(device)
                rollouts.share_memory_()
            return rollouts

        self._agent.post_init(create_ver_rollouts_fn)

        main_is_iw = not self.ver_config.overlap_rollouts_and_learn
        n_inference_workers = self.ver_config.num_inference_workers

        with inference_mode():
            if self.ver_config.overlap_rollouts_and_learn:
                self.learning_rollouts = VERRolloutStorage(**storage_kwargs)
                self.learning_rollouts.to(self.device)
            else:
                self.learning_rollouts = self._agent.rollouts

            storage_kwargs["observation_space"] = actor_obs_space
            storage_kwargs["numsteps"] = 1

            self._transfer_buffers = (
                VERRolloutStorage(**storage_kwargs)
                .buffers.slice_keys(
                    "rewards",
                    "masks",
                    "observations",
                    "episode_ids",
                    "environment_ids",
                    "actions",
                    "step_ids",
                )
                .map_in_place(lambda t: t.share_memory_())
            )[
                slice(0, len(self.environment_workers))
                if self.ver_config.variable_experience
                else 0
            ]

        self._agent.actor_critic.share_memory()

        if self._is_distributed:
            self._agent.updater.init_distributed(find_unused_params=False)  # type: ignore[operator]

        self._iw_sync = InferenceWorkerSync(
            self.mp_ctx,
            n_inference_workers,
        )

        inference_worker_args = (
            n_inference_workers,
            self.config,
            self.queues,
            self._iw_sync,
            self._transfer_buffers,
            self.config.habitat_baselines.rl.policy.main_agent.name,
            (
                self.config,
                self._env_spec.observation_space,
                self._env_spec.action_space,
            ),
            self.device,
            self.preemption_decider.rollout_ends,
        )

        self._transfer_policy_tensors = list(
            self._agent.actor_critic.all_policy_tensors()
        )

        self.inference_workers = [
            InferenceWorker(self.mp_ctx, i, *inference_worker_args)
            for i in range(1 if main_is_iw else 0, n_inference_workers)
        ]
        if main_is_iw:
            self._inference_worker_impl = InferenceWorkerProcess(
                None,
                None,
                None,
                0,
                *inference_worker_args,
            )
            self._inference_worker_impl.set_actor_critic_tensors(
                self._transfer_policy_tensors
            )
            self._inference_worker_impl.set_rollouts(self._agent.rollouts)
        else:
            self._inference_worker_impl = None

        for iw in self.inference_workers:
            # We send the policy weights and the rollouts
            # via a torch.multiprocessing.SimpleQueue instead
            # of in the constructor as otherwise the shared
            # cuda tensors don't get properly freed on
            # destruction which causes an error.
            iw.set_actor_critic_tensors(self._transfer_policy_tensors)
            iw.set_rollouts(self._agent.rollouts)
            iw.start()

        ews_to_wait = []
        for i, ew in enumerate(self.environment_workers):
            ew.set_transfer_buffers(self._transfer_buffers)
            if i > 0:
                init_reports.append(ew.get_init_report())

            ew.wait_start()
            ews_to_wait.append(ew)
            if len(ews_to_wait) >= 4:
                [a.wait_sync() for a in ews_to_wait]
                ews_to_wait = []

        [a.wait_sync() for a in ews_to_wait]

        if self._is_distributed:
            torch.distributed.barrier()
        [aw.start_experience_collection() for aw in self.environment_workers]
        self.report_worker.start_collection()

        self.timer = Timing()

        self._all_workers.extend(self.environment_workers)
        self._all_workers.extend(self.inference_workers)
        self._all_workers.append(self.report_worker)
        self._all_workers.append(self.preemption_decider)

    @profiling_wrapper.RangeContext("_update_agent")
    def _update_agent(self):
        torch.backends.cudnn.benchmark = True
        ppo_cfg = self.config.habitat_baselines.rl.ppo

        with self.timer.avg_time("learn"):
            t_compute_returns = time.perf_counter()

            with self.timer.avg_time("compute returns"), inference_mode():
                self.learning_rollouts.compute_returns(
                    ppo_cfg.use_gae,
                    ppo_cfg.gamma,
                    ppo_cfg.tau,
                )

            compute_returns_time = time.perf_counter() - t_compute_returns

            if self._is_distributed:
                with self.timer.avg_time("synchronize"):
                    torch.distributed.barrier()

            t_update_model = time.perf_counter()
            with self.timer.avg_time("update agent"):
                self._agent.updater.train()

                losses = self._agent.updater.update(self.learning_rollouts)

            with self.timer.avg_time("after update"), inference_mode():
                for t_src, t_dst in zip(
                    self._agent.actor_critic.all_policy_tensors(),
                    self._transfer_policy_tensors,
                ):
                    # Some torch modules update their buffers out of place.
                    # So we may need to copy the new value into the transfer
                    # buffers.
                    if t_src is not t_dst:
                        t_dst.copy_(t_src)

                if not self.ver_config.overlap_rollouts_and_learn:
                    self.learning_rollouts.after_update()
                    self._iw_sync.should_start_next.set()

                self._agent.rollouts.increment_policy_version()

        self._learning_time = (
            time.perf_counter() - t_update_model
        ) + compute_returns_time
        torch.backends.cudnn.benchmark = False
        self._agent.after_update()

        return losses

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training DD/PPO.

        Returns:
            None
        """
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        self.num_steps_done = 0
        resume_state = load_resume_state(self.config)
        if resume_state is not None:
            if not self.config.habitat_baselines.load_resume_state_config:
                raise FileExistsError(
                    f"The configuration provided has habitat_baselines.load_resume_state_config=False but a previous training run exists. You can either delete the checkpoint folder {self.config.habitat_baselines.checkpoint_folder}, or change the configuration key habitat_baselines.checkpoint_folder in your new run."
                )

            self.config = self._get_resume_state_config_or_new_config(
                resume_state["config"]
            )

            requeue_stats = resume_state["requeue_stats"]
            self.num_steps_done = requeue_stats["num_steps_done"]
            self.num_updates_done = requeue_stats["num_updates_done"]

        self._init_train(resume_state)

        count_checkpoints = 0

        if resume_state is not None:
            requeue_stats = resume_state["requeue_stats"]
            self._last_checkpoint_percent = requeue_stats[
                "_last_checkpoint_percent"
            ]
            count_checkpoints = requeue_stats["count_checkpoints"]

        if self.ver_config.overlap_rollouts_and_learn:
            self.preemption_decider.start_rollout()

        while not self.is_done():
            profiling_wrapper.on_start_step()

            self._agent.pre_rollout()

            if rank0_only() and self._should_save_resume_state():
                requeue_stats = dict(
                    count_checkpoints=count_checkpoints,
                    num_steps_done=self.num_steps_done,
                    num_updates_done=self.num_updates_done,
                    _last_checkpoint_percent=self._last_checkpoint_percent,
                    report_worker_state=self.report_worker.state_dict(),
                )
                resume_state = {
                    **self._agent.get_resume_state(),
                    "config": self.config,
                    "requeue_stats": requeue_stats,
                }

                save_resume_state(
                    resume_state,
                    self.config,
                )

            if EXIT.is_set():
                profiling_wrapper.range_pop()  # train update
                [w.close() for w in self._all_workers]
                [w.join() for w in self._all_workers]

                requeue_job()
                break

            with inference_mode():
                if not self.ver_config.overlap_rollouts_and_learn:
                    self.preemption_decider.start_rollout()
                    while not self._agent.rollouts.rollout_done:
                        self._inference_worker_impl.try_one_step()

                    self._inference_worker_impl.finish_rollout()

                self._iw_sync.rollout_done.wait()
                self._iw_sync.rollout_done.clear()

                if self._iw_sync.all_workers.n_waiting > 0:
                    raise RuntimeError(
                        f"{self._iw_sync.all_workers.n_waiting} inference worker(s) is(are) still waiting on the IW barrier. "
                        "Likely they never waited on it.\n"
                    )

                self._agent.rollouts.after_rollout()

                if self.ver_config.overlap_rollouts_and_learn:
                    with self.timer.avg_time("overlap_transfers"):
                        self.learning_rollouts.copy(self._agent.rollouts)

                self.preemption_decider.end_rollout(
                    self._agent.rollouts.num_steps_to_collect
                )

                self.queues.report.put(
                    (
                        ReportWorkerTasks.num_steps_collected,
                        int(self._agent.rollouts.num_steps_collected),
                    )
                )

                if self.ver_config.overlap_rollouts_and_learn:
                    with self.timer.avg_time("overlap_transfers"):
                        self._agent.rollouts.after_update()
                        self._iw_sync.should_start_next.set()
                        self.preemption_decider.start_rollout()

            losses = self._update_agent()

            self.preemption_decider.learner_time(self._learning_time)

            self.queues.report.put_many(
                (
                    (
                        ReportWorkerTasks.learner_timing,
                        self.timer,
                    ),
                    (
                        ReportWorkerTasks.learner_update,
                        losses,
                    ),
                )
            )
            self.timer = Timing()

            self.num_steps_done = int(self.report_worker.num_steps_done)

            self.num_updates_done += 1
            # checkpoint model
            if rank0_only() and self.should_checkpoint():
                self.save_checkpoint(
                    f"ckpt.{count_checkpoints}.pth",
                    dict(
                        step=self.num_steps_done,
                        wall_time=self.report_worker.time_taken,
                    ),
                )
                count_checkpoints += 1

        self.window_episode_stats = (
            self.report_worker.get_window_episode_stats()
        )

        [w.close() for w in self._all_workers]
        [w.join() for w in self._all_workers]

        if self._is_distributed:
            torch.distributed.barrier()
