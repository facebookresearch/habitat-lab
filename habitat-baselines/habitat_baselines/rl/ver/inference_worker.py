#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import queue
import time
from multiprocessing import SimpleQueue
from multiprocessing.context import BaseContext
from typing import TYPE_CHECKING, List, Optional, Tuple

import attr
import numpy as np
import torch

from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensor_dict import (
    NDArrayDict,
    TensorDict,
    iterate_dicts_recursively,
)
from habitat_baselines.common.windowed_running_mean import WindowedRunningMean
from habitat_baselines.rl.ddppo.policy.resnet_policy import PointNavResNetNet
from habitat_baselines.rl.ppo.policy import NetPolicy
from habitat_baselines.rl.ver.task_enums import (
    EnvironmentWorkerTasks,
    InferenceWorkerTasks,
    PreemptionDeciderTasks,
    ReportWorkerTasks,
)
from habitat_baselines.rl.ver.ver_rollout_storage import VERRolloutStorage
from habitat_baselines.rl.ver.worker_common import (
    InferenceWorkerSync,
    ProcessBase,
    RolloutEarlyEnds,
    WorkerBase,
    WorkerQueues,
)
from habitat_baselines.utils.common import batch_obs, inference_mode
from habitat_baselines.utils.timing import Timing

if TYPE_CHECKING:
    from omegaconf import DictConfig


@attr.s(auto_attribs=True)
class InferenceWorkerProcess(ProcessBase):
    setup_queue: SimpleQueue
    inference_worker_idx: int
    num_inference_workers: int
    config: "DictConfig"
    queues: WorkerQueues
    iw_sync: InferenceWorkerSync
    _torch_transfer_buffers: TensorDict
    policy_name: str
    policy_args: Tuple
    device: torch.device
    rollout_ends: RolloutEarlyEnds
    actor_critic_tensors: List[torch.Tensor] = attr.ib(None, init=False)
    rollouts: VERRolloutStorage = attr.ib(None, init=False)
    replay_reqs: List = attr.ib(factory=list, init=False)
    new_reqs: List = attr.ib(factory=list, init=False)
    _avg_step_time: WindowedRunningMean = attr.ib(
        factory=lambda: WindowedRunningMean(128), init=False
    )
    timer: Timing = attr.Factory(Timing)
    _n_replay_steps: int = 0
    actor_critic: NetPolicy = attr.ib(init=False)
    visual_encoder: Optional[torch.nn.Module] = attr.ib(
        init=False, default=None
    )
    _static_encoder: bool = attr.ib(init=False, default=False)
    transfer_buffers: NDArrayDict = attr.ib(default=None, init=False)
    incoming_transfer_buffers: NDArrayDict = attr.ib(default=None, init=False)

    def __attrs_post_init__(self):
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self._overlapped = (
            self.config.habitat_baselines.rl.ver.overlap_rollouts_and_learn
        )
        with inference_mode():
            self.actor_critic = baseline_registry.get_policy(
                self.policy_name
            ).from_config(*self.policy_args)
            self.actor_critic.eval()
            self.actor_critic.aux_loss_modules.clear()
            self.actor_critic.to(device=self.device)
            if not self.config.habitat_baselines.rl.ddppo.train_encoder:
                self._static_encoder = True
                self.visual_encoder = self.actor_critic.net.visual_encoder

        self.transfer_buffers = self._torch_transfer_buffers.numpy()
        self.incoming_transfer_buffers = self.transfer_buffers.slice_keys(
            set(self.transfer_buffers.keys()) - {"actions"}
        )
        self.last_step_time = time.perf_counter()
        self.min_reqs = int(
            max(
                len(self.queues.environments)
                / self.num_inference_workers
                / 1.5,
                1,
            )
        )
        self.max_reqs = int(
            max(
                len(self.queues.environments)
                / self.num_inference_workers
                * 1.5,
                1,
            )
        )
        assert self.max_reqs >= self.min_reqs

        self.min_wait_time = 0.01
        self.obs_transforms = get_active_obs_transforms(self.config)
        self._variable_experience = (
            self.config.habitat_baselines.rl.ver.variable_experience
        )

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"InferenceWorker-{self.inference_worker_idx} initialized")

    def set_actor_critic_tensors(self, actor_critic_tensors):
        with inference_mode():
            self.actor_critic_tensors = actor_critic_tensors
            if not self._overlapped:
                # If learning and experience collection aren't overlapped,
                # we can share the weights between all the inference workers and
                # the learner.
                for i, t in enumerate(self.actor_critic.all_policy_tensors()):
                    t.set_(self.actor_critic_tensors[i])
            else:
                # Otherwise each inference worker needs its own copy.
                self._update_actor_critic()

    def set_rollouts(self, rollouts: VERRolloutStorage):
        self.rollouts = rollouts
        self._current_policy_version = int(
            self.rollouts.cpu_current_policy_version
        )

    def _update_actor_critic(self):
        for src, dst in zip(
            self.actor_critic_tensors, self.actor_critic.all_policy_tensors()
        ):
            dst.copy_(src)

    def _update_storage_no_ver(
        self, prev_step: TensorDict, current_step: TensorDict, current_steps
    ):
        for offset, step_content in (
            (-1, prev_step),
            (0, current_step),
        ):
            step_plus_off = current_steps + offset

            for src, dst in iterate_dicts_recursively(
                step_content,
                self.rollouts.buffers.slice_keys(step_content.keys()),
            ):
                ##
                # NB: This loop is faster than something fancy with index_copy_
                # or gather_!  I think that's because copy_ is much faster and
                # async
                for i, env_idx in enumerate(self.new_reqs):
                    step_idx = step_plus_off[env_idx].item()
                    if 0 <= step_idx <= self.rollouts.num_steps:
                        dst[step_idx, env_idx].copy_(src[i])

    def _update_storage_ver(
        self, prev_step: TensorDict, current_step: TensorDict, my_slice
    ):
        for src, dst in iterate_dicts_recursively(
            prev_step, self.rollouts.buffers.slice_keys(prev_step.keys())
        ):
            ##
            # NB: This loop is faster than something fancy with index_copy_
            # or gather_!  I think that's because copy_ is much faster and
            # async
            for i, env_idx in enumerate(self.new_reqs):
                dst_idx = self._prev_inds[env_idx].item()
                if dst_idx >= 0:
                    dst[dst_idx].copy_(src[i])

        assert torch.all(self.rollouts.buffers["is_stale"][my_slice])

        self.rollouts.buffers.slice_keys(current_step.keys())[
            my_slice
        ] = current_step

    def _sync_device(self):
        if self.num_inference_workers > 1 and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextlib.contextmanager
    def lock_and_sync(self):
        if self.num_inference_workers == 1:
            yield
            return

        try:
            with self.iw_sync.lock:
                yield

                self._sync_device()
        finally:
            pass

    def step(self) -> Tuple[bool, List[Tuple[int, int]]]:
        steps_finished: List[Tuple[int, int]] = []
        if len(self.new_reqs) == 0:
            return False, steps_finished

        if (
            self._current_policy_version
            != self.rollouts.cpu_current_policy_version
        ):
            self._current_policy_version = int(
                self.rollouts.cpu_current_policy_version
            )
            if self._overlapped:
                self._update_actor_critic()

        current_steps = self.rollouts.current_steps.copy()
        final_batch = False

        if self._variable_experience:
            self.new_reqs.sort(
                key=lambda a: (
                    self.rollouts.actor_steps_collected[a],
                    a,
                )
            )
            with self.lock_and_sync():
                ptr = self.rollouts.ptr.item()
                num_to_process = int(
                    min(
                        int(
                            self.rollouts.num_steps_to_collect
                            - self.rollouts.num_steps_collected
                        ),
                        len(self.new_reqs),
                    )
                )
                next_ptr = ptr + num_to_process
                assert next_ptr <= self.rollouts.buffer_size
                self.rollouts.ptr[:] = next_ptr
                self.rollouts.num_steps_collected += (
                    num_to_process - self._n_replay_steps
                )

                if (
                    self.rollouts.num_steps_collected
                ) == self.rollouts.num_steps_to_collect:
                    final_batch = True
                    self.rollouts.rollout_done[:] = True

            my_slice = slice(ptr, next_ptr)
            self.replay_reqs += self.new_reqs[num_to_process:]
            self.new_reqs = self.new_reqs[:num_to_process]
        else:
            for r in self.new_reqs:
                if current_steps[r] > self.rollouts.num_steps:
                    raise RuntimeError(
                        f"Got a step from actor {r} after collecting {current_steps[r]} steps. This shouldn't be possible."
                    )

            if len(self.new_reqs) > 0:
                with self.lock_and_sync():
                    self.rollouts.num_steps_collected += (
                        len(self.new_reqs) - self._n_replay_steps
                    )

                    if (
                        self.rollouts.num_steps_collected
                        == self.rollouts.num_steps_to_collect
                    ):
                        final_batch = True
                        self.rollouts.rollout_done[:] = True

        if len(self.new_reqs) == 0:
            return False, steps_finished

        if self._n_replay_steps > 0 and len(self.replay_reqs) > 0:
            raise RuntimeError(
                f"Added to replay reqs before reqs from the last rollout were replayed. {self.replay_reqs}"
            )

        with self.timer.avg_time("batch obs"):
            to_batch = [
                self.incoming_transfer_buffers[env_idx]
                for env_idx in self.new_reqs
            ]

            to_batch = batch_obs(to_batch, device=self.device)
            obs = to_batch.pop("observations")

            environment_ids = to_batch["environment_ids"].view(-1)

        with self.timer.avg_time("step policy"):
            obs = apply_obs_transforms_batch(obs, self.obs_transforms)
            recurrent_hidden_states = self.rollouts.next_hidden_states[
                environment_ids
            ]
            prev_actions = self.rollouts.next_prev_actions[environment_ids]
            if self._static_encoder:
                obs[
                    PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                ] = self.visual_encoder(obs)

            action_data = self.actor_critic.act(
                obs,
                recurrent_hidden_states,
                prev_actions,
                to_batch["masks"],
            )

            if not final_batch:
                self.rollouts.next_hidden_states.index_copy_(
                    0,
                    environment_ids,
                    action_data.rnn_hidden_states,
                )
                self.rollouts.next_prev_actions.index_copy_(
                    0, environment_ids, action_data.actions
                )

            if self._variable_experience:
                self._prev_inds = self.rollouts.prev_inds.copy()

                self.rollouts.prev_inds[self.new_reqs] = np.arange(
                    my_slice.start,
                    my_slice.stop,
                    dtype=np.int64,
                )

            cpu_actions = action_data.env_actions.to(device="cpu")
            self.transfer_buffers["actions"][
                self.new_reqs
            ] = cpu_actions.numpy()

            self._sync_device()

            for env_idx in self.new_reqs:
                steps_finished.append(
                    (
                        int(self.rollouts.current_steps[env_idx]),
                        int(env_idx),
                    )
                )
                self.rollouts.actor_steps_collected[env_idx] += 1
                self.rollouts.current_steps[env_idx] += 1

                if self._variable_experience:
                    final_step = final_batch
                else:
                    final_step = self.rollouts.current_steps[env_idx] == (
                        self.rollouts.num_steps + 1
                    )

                if not final_step:
                    self.queues.environments[env_idx].put(
                        (EnvironmentWorkerTasks.step, None)
                    )
                else:
                    # We 'replay' the steps in the final
                    # batch of experience collected by the policy
                    # worker. This allows us to use the inference worker
                    # to compute the V-est for bootstrapping the returns
                    self.replay_reqs.append(env_idx)

        with self.timer.avg_time("update storage"):
            current_step = TensorDict.from_tree(
                dict(
                    masks=to_batch["masks"],
                    observations=obs,
                    actions=action_data.actions,
                    action_log_probs=action_data.action_log_probs,
                    recurrent_hidden_states=recurrent_hidden_states,
                    prev_actions=prev_actions,
                    policy_version=self.rollouts.current_policy_version.expand(
                        len(self.new_reqs), 1
                    ),
                    episode_ids=to_batch["episode_ids"],
                    environment_ids=to_batch["environment_ids"],
                    step_ids=to_batch["step_ids"],
                    value_preds=action_data.values,
                    returns=torch.full(
                        (),
                        float("nan"),
                        device=self.device,
                    ).expand(len(self.new_reqs), 1),
                )
            )

            prev_step = TensorDict.from_tree(dict(rewards=to_batch["rewards"]))

            if self._variable_experience:
                self._update_storage_ver(prev_step, current_step, my_slice)
            else:
                self._update_storage_no_ver(
                    prev_step, current_step, current_steps
                )

        self.new_reqs = []

        return True, steps_finished

    def finish_rollout(self):
        with self.timer.add_time("rollout"), self.timer.avg_time(
            "finish rollout"
        ):
            self._sync_device()

            # First barrier wait makes sure we don't put outstanding or replay reqs into the
            # queue before everyone is done
            self.iw_sync.all_workers.wait()
            self.queues.inference.put_many(self.replay_reqs + self.new_reqs)
            self.replay_reqs = []
            self.new_reqs = []

            # Second wait makes sure we don't read from the queue before
            # everyone has put their outstanding or replay reqs into the queue
            self.iw_sync.all_workers.wait()

            self.iw_sync.should_start_next.clear()

            # Give the replay steps to the last inference worker as this
            # one is guaranteed to not be the main process (when there's more than 1)
            if self.inference_worker_idx == (
                self.config.habitat_baselines.rl.ver.num_inference_workers - 1
            ):
                while not self.queues.inference.empty():
                    self.new_reqs += self.queues.inference.get_many()

                self._n_replay_steps = len(self.new_reqs)
                self.rollouts.will_replay_step[self.new_reqs] = True

                self.iw_sync.rollout_done.set()

        self.queues.report.put((ReportWorkerTasks.policy_timing, self.timer))
        self.timer = Timing()

    def _get_more_reqs(self) -> List[int]:
        if len(self.new_reqs) >= self.max_reqs:
            return []

        try:
            with self.timer.add_time("wait"):
                return self.queues.inference.get_many(
                    timeout=0.005,
                    max_messages_to_get=self.max_reqs - len(self.new_reqs),
                )
        except queue.Empty:
            return []

    def try_one_step(self):
        with self.timer.add_time("rollout"):
            stepped = False

            self.new_reqs += self._get_more_reqs()

            should_try_step = len(self.new_reqs) > 0 and (
                len(self.new_reqs) >= self.min_reqs
                or (time.perf_counter() - self.last_step_time)
                > self.min_wait_time
            )

            if should_try_step:
                t_step_start = time.perf_counter()
                stepped, steps_finished = self.step()
                t_step_end = time.perf_counter()

                if stepped:
                    self.queues.preemption_decider.put(
                        (
                            PreemptionDeciderTasks.policy_step,
                            dict(
                                worker_idx=self.inference_worker_idx,
                                steps_finished=steps_finished,
                                t_stamp=t_step_end,
                            ),
                        )
                    )
                    self._avg_step_time.add(t_step_end - t_step_start)
                    self.last_step_time = t_step_end

                    self.min_wait_time = self._avg_step_time.mean / 2
                    self._n_replay_steps = 0

            return stepped

    @inference_mode()
    def run(self):
        assert self.done_event is not None
        self.actor_critic.eval()

        task = None
        while task != InferenceWorkerTasks.start:
            task, data = self.setup_queue.get()
            if task == InferenceWorkerTasks.set_actor_critic_tensors:
                self.set_actor_critic_tensors(data)
            elif task == InferenceWorkerTasks.set_rollouts:
                self.set_rollouts(data)
            elif task == InferenceWorkerTasks.start:
                break
            else:
                raise RuntimeError(f"IW Unknown task: {task}")

        while not self.done_event.is_set():
            if not self.iw_sync.should_start_next.is_set():
                self.iw_sync.should_start_next.wait(timeout=1.0)
            else:
                self.try_one_step()

                self.update_should_end_early()

                if self.rollouts.rollout_done:
                    self.finish_rollout()

    def update_should_end_early(self) -> None:
        if not self._variable_experience:
            return

        use_time = True
        if use_time:
            if self.rollout_ends.time.value < 0.0:
                return

            end_early = self.rollout_ends.time.value <= time.perf_counter()

            if end_early:
                with self.iw_sync.lock:
                    self.rollouts.rollout_done[:] = True
        else:
            if self.rollout_ends.steps.value < 0.0:
                return
            with self.iw_sync.lock:
                self.rollouts.rollout_done[:] = bool(
                    self.rollouts.rollout_done
                ) or (
                    self.rollout_ends.steps.value
                    <= self.rollouts.num_steps_collected
                )


class InferenceWorker(WorkerBase):
    def __init__(self, mp_ctx: BaseContext, *args, **kwargs):
        self.setup_queue = mp_ctx.SimpleQueue()
        super().__init__(
            mp_ctx, InferenceWorkerProcess, self.setup_queue, *args, **kwargs
        )

    def set_actor_critic_tensors(self, actor_critic_tensors):
        self.setup_queue.put(
            (
                InferenceWorkerTasks.set_actor_critic_tensors,
                actor_critic_tensors,
            )
        )

    def set_rollouts(self, rollouts):
        self.setup_queue.put((InferenceWorkerTasks.set_rollouts, rollouts))

    def start(self):
        self.setup_queue.put((InferenceWorkerTasks.start, None))
