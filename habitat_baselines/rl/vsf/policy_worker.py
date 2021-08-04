import contextlib
import queue
import time
from multiprocessing import Event
from multiprocessing.context import BaseContext
from typing import List, Optional

import attr
import faster_fifo
import numpy as np
import torch

from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.rl.vsf.task_enums import (
    ActorWorkerTasks,
    ReportWorkerTasks,
)
from habitat_baselines.rl.vsf.timing import Timing
from habitat_baselines.rl.vsf.vsf_rollout_storage import VSFRolloutStorage
from habitat_baselines.rl.vsf.worker_common import ProcessBase, WorkerBase
from habitat_baselines.utils.common import ObservationBatchingCache, batch_obs


@contextlib.contextmanager
def _set_stream(s):
    try:
        with torch.cuda.stream(s) if s is not None else contextlib.suppress():
            yield
    finally:
        pass


@attr.s(auto_attribs=True)
class _RunningMean:
    _sum: float = 0.0
    _count: int = 0

    def add(self, v):
        self._sum += v
        self._count += 1

    @property
    def mean(self):
        return self._sum / max(self._count, 1)


@attr.s(auto_attribs=True)
class PolicyWorkerProcess(ProcessBase):
    policy_worker_idx: int
    policy_worker_queue: faster_fifo.Queue
    actor_worker_queues: List[faster_fifo.Queue]
    report_queue: faster_fifo.Queue
    rollout_storage: VSFRolloutStorage
    transfer_buffers: TensorDict
    actor_critic: Policy
    device: torch.device
    done_event: Optional[Event] = None
    batching_cache: ObservationBatchingCache = attr.ib(
        factory=ObservationBatchingCache, init=False
    )
    finished_reqs: List = attr.ib(factory=list, init=False)
    new_reqs: List = attr.ib(factory=list, init=False)
    _avg_step_time: _RunningMean = attr.ib(factory=_RunningMean, init=False)
    timer: Timing = attr.Factory(Timing)
    finish_ratio: float = 0.0
    _first_rollout: bool = True

    def __attrs_post_init__(self):
        self.transfer_buffers.map_in_place(lambda t: t.numpy())
        self.last_step_time = time.time()
        self.min_reqs = max(len(self.actor_worker_queues) // 4, 1)
        self.min_wait_time = self._avg_step_time.mean / 2

    def step(self):
        if len(self.new_reqs) == 0:
            return False

        with self.timer.avg_time("batch obs"):
            cpu_steps = self.rollout_storage.current_steps.to(
                device="cpu", non_blocking=True
            )
            actor_inds = [
                np.array(actor_idx, dtype=np.int64)
                for actor_idx in self.new_reqs
            ]

            obs = [
                self.transfer_buffers["observations"][actor_idx]
                for actor_idx in self.new_reqs
            ]
            rewards = [
                self.transfer_buffers["rewards"][actor_idx]
                for actor_idx in self.new_reqs
            ]
            not_dones = [
                self.transfer_buffers["masks"][actor_idx]
                for actor_idx in self.new_reqs
            ]

            to_batch = obs
            for b, r, d, ind in zip(to_batch, rewards, not_dones, actor_inds):
                b["_/not_dones"] = d
                b["_/actor_inds"] = ind
                b["_/rewards"] = r

            to_batch = batch_obs(
                to_batch, device=self.device, cache=self.batching_cache
            )
            obs = {k: to_batch[k] for k in obs[0].keys()}

            actor_inds = to_batch["_/actor_inds"]

        with self.timer.avg_time("step policy"):
            current_steps = self.rollout_storage.current_steps[
                actor_inds
            ].clone()

            recurrent_hidden_states = self.rollout_storage.next_hidden_states[
                actor_inds
            ]
            prev_actions = self.rollout_storage.next_prev_actions[actor_inds]

            (
                values,
                actions,
                actions_log_probs,
                next_recurrent_hidden_states,
            ) = self.actor_critic.act(
                obs,
                recurrent_hidden_states,
                prev_actions,
                to_batch["_/not_dones"],
            )

            self.rollout_storage.next_hidden_states.index_copy_(
                0, actor_inds, next_recurrent_hidden_states
            )
            self.rollout_storage.next_prev_actions.index_copy_(
                0, actor_inds, actions
            )
            self.rollout_storage.current_steps[actor_inds] += 1

            cpu_actions = actions.to(device="cpu")

            n_done = 0
            for actor_idx, action in zip(self.new_reqs, cpu_actions.unbind(0)):
                if cpu_steps[actor_idx] < self.rollout_storage.numsteps:
                    self.actor_worker_queues[actor_idx].put(
                        (ActorWorkerTasks.step, action)
                    )
                else:
                    self.finished_reqs.append(actor_idx)
                    n_done += 1

            with self.rollout_storage.lock:
                self.rollout_storage.n_actors_done += n_done

                if self.rollout_storage.n_actors_done >= (
                    self.rollout_storage._num_envs
                    if self._first_rollout
                    else max(
                        self.finish_ratio * self.rollout_storage._num_envs, 1.0
                    )
                ):
                    self.rollout_storage.rollout_done[:] = True

        with self.timer.avg_time("update storage"):
            current_step = dict(
                masks=to_batch["_/not_dones"],
                observations=obs,
                actions=actions,
                action_log_probs=actions_log_probs,
                value_preds=values,
                recurrent_hidden_states=recurrent_hidden_states,
                prev_actions=prev_actions,
            )
            prev_step = dict(rewards=to_batch["_/rewards"])

            flat_buffers = self.rollout_storage.buffers.map(
                lambda t: t.view(-1, *t.size()[2:])
            )
            for offset, step_content in (
                (-1, prev_step),
                (0, current_step),
            ):
                step_plus_off = current_steps + offset
                mask = (step_plus_off >= 0) & (
                    step_plus_off <= self.rollout_storage.numsteps
                )

                indexer = (
                    step_plus_off * self.rollout_storage._num_envs + actor_inds
                )
                flat_buffers.zip(step_content, strict=False).map(
                    lambda dst, src: dst.index_copy_(
                        0, indexer[mask], src[mask]
                    )
                )

        self.new_reqs = []

        return True

    def finish_rollout(self):
        self.report_queue.put((ReportWorkerTasks.policy_timing, self.timer))
        self.timer = Timing()
        self.rollout_storage.all_done.wait()
        self.policy_worker_queue.put_many(self.finished_reqs + self.new_reqs)
        self.finished_reqs = []
        self.new_reqs = []
        self.rollout_storage.storage_free.clear()
        self._first_rollout = False

    def try_one_step(self):
        stepped = False

        try:
            self.new_reqs += self.policy_worker_queue.get_many(
                timeout=max(
                    self.min_wait_time - (time.time() - self.last_step_time), 0
                )
            )
        except queue.Empty:
            pass

        if (
            len(self.new_reqs) >= self.min_reqs
            or (time.time() - self.last_step_time) > self.min_wait_time
        ):
            t_step_start = time.time()
            stepped = self.step()
            t_step_end = time.time()

            if stepped:
                self._avg_step_time.add(t_step_end - t_step_start)
                self.last_step_time = t_step_end

                self.min_wait_time = self._avg_step_time.mean / 2

        return stepped

    def run(self):
        assert self.done_event is not None
        self.actor_critic.eval()

        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        while not self.done_event.is_set():
            if not self.rollout_storage.storage_free.is_set():
                self.rollout_storage.storage_free.wait(timeout=1.0)

            else:
                self.try_one_step()

                if self.rollout_storage.rollout_done:
                    self.finish_rollout()


class PolicyWorker(WorkerBase):
    def __init__(
        self,
        mp_ctx: BaseContext,
        policy_worker_idx: int,
        policy_worker_queue: faster_fifo.Queue,
        actor_worker_queues: List[faster_fifo.Queue],
        report_queue: faster_fifo.Queue,
        rollout_storage: VSFRolloutStorage,
        transfer_buffers: TensorDict,
        actor_critic: Policy,
        device,
    ):
        super().__init__(
            mp_ctx,
            PolicyWorkerProcess,
            policy_worker_idx,
            policy_worker_queue,
            actor_worker_queues,
            report_queue,
            rollout_storage,
            transfer_buffers,
            actor_critic,
            device,
        )
