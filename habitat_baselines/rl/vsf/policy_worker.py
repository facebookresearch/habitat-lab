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


@attr.s(auto_attribs=True)
class RunningMean:
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
    rollout_storage: VSFRolloutStorage
    transfer_buffers: TensorDict
    actor_critic: Policy
    device: torch.device
    done_event: Optional[Event] = None
    batching_cache: ObservationBatchingCache = attr.ib(
        factory=ObservationBatchingCache, init=False
    )
    reqs: List = attr.ib(factory=list, init=False)
    _avg_step_time: RunningMean = attr.ib(factory=RunningMean, init=False)
    timer: Timing = attr.Factory(Timing)

    def step(self):
        reqs_to_process = []
        reqs_remaining = []
        actor_inds = []

        with self.rollout_storage.lock:
            cpu_steps = self.rollout_storage.current_steps.to(
                device="cpu", non_blocking=True
            )
        for actor_idx in self.reqs:
            if cpu_steps[actor_idx] <= self.rollout_storage.numsteps:
                reqs_to_process.append(actor_idx)
                actor_inds.append(np.array(actor_idx, dtype=np.int64))
            else:
                reqs_remaining.append(actor_idx)

        if len(reqs_to_process) == 0:
            return (
                self.rollout_storage.n_actors_done
                == self.rollout_storage._num_envs
            )

        with self.timer.avg_time("Batch-Obs"):
            obs = [
                self.transfer_buffers["observations"][actor_idx]
                for actor_idx in reqs_to_process
            ]
            rewards = [
                self.transfer_buffers["rewards"][actor_idx]
                for actor_idx in reqs_to_process
            ]
            not_dones = [
                self.transfer_buffers["masks"][actor_idx]
                for actor_idx in reqs_to_process
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

        with self.timer.avg_time("Step-Policy"):
            current_steps = self.rollout_storage.current_steps[actor_inds]

            hidden_states = self.rollout_storage.buffers[
                "recurrent_hidden_states"
            ][current_steps, actor_inds]
            prev_actions = self.rollout_storage.buffers["prev_actions"][
                current_steps, actor_inds
            ]

            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
            ) = self.actor_critic.act(
                obs,
                hidden_states,
                prev_actions,
                to_batch["_/not_dones"],
            )

            next_step = dict(
                recurrent_hidden_states=recurrent_hidden_states,
                prev_actions=actions,
            )

            current_step = dict(
                masks=to_batch["_/not_dones"],
                observations=obs,
                actions=actions,
                action_log_probs=actions_log_probs,
                value_preds=values,
            )
            prev_step = dict(rewards=to_batch["_/rewards"])

            flat_buffers = self.rollout_storage.buffers.map(
                lambda t: t.view(-1, *t.size()[2:])
            )
            for offset, step_content in (
                (-1, prev_step),
                (0, current_step),
                (1, next_step),
            ):
                mask = ((current_steps + offset) >= 0) & (
                    (current_steps + offset) <= self.rollout_storage.numsteps
                )

                indexer = (
                    current_steps + offset
                ) * self.rollout_storage._num_envs + actor_inds
                flat_buffers.zip(step_content, strict=False).map(
                    lambda dst, src: dst.index_copy_(
                        0, indexer[mask], src[mask]
                    )
                )

            cpu_actions = actions.to(device="cpu")

        n_done = 0
        for actor_idx, action in zip(reqs_to_process, cpu_actions.unbind(0)):
            if cpu_steps[actor_idx] < self.rollout_storage.numsteps:
                self.actor_worker_queues[actor_idx].put(
                    (ActorWorkerTasks.step, action)
                )
            else:
                reqs_remaining.append(actor_idx)
                n_done += 1

        self.reqs = reqs_remaining
        with self.rollout_storage.lock:
            self.rollout_storage.current_steps[actor_inds] += 1
            self.rollout_storage.n_actors_done += n_done
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

        return (
            self.rollout_storage.n_actors_done
            == self.rollout_storage._num_envs
        )

    def run_one_epoch(self):
        last_step_time = time.time()
        min_reqs = max(len(self.actor_worker_queues) // 4, 1)
        min_wait_time = self._avg_step_time.mean / 2

        new_reqs = []
        while self.done_event is None or not self.done_event.is_set():
            try:
                new_reqs += self.policy_worker_queue.get_many(
                    timeout=max(
                        min_wait_time - (time.time() - last_step_time), 0
                    )
                )
            except queue.Empty:
                pass

            if (
                len(new_reqs) >= min_reqs
                or (time.time() - last_step_time) > min_wait_time
            ):
                self.reqs.extend(new_reqs)
                new_reqs = []
                t_step_start = time.time()
                is_done = self.step()
                if is_done:
                    self.report_queue.put(
                        (ReportWorkerTasks.policy_timing, self.timer)
                    )
                    self.rollout_storage.all_done.wait()
                    return

                t_step_end = time.time()
                self._avg_step_time.add(t_step_end - t_step_start)
                last_step_time = t_step_end

                min_wait_time = self._avg_step_time.mean / 2

    def run(self):
        self.actor_critic.eval()
        while not self.done_event.is_set():
            if not self.rollout_storage.storage_free.is_set():
                self.rollout_storage.storage_free.wait(timeout=1.0)
            else:
                self.run_one_epoch()


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
    ):
        super().__init__(
            mp_ctx,
            PolicyWorkerProcess,
            policy_worker_idx,
            policy_worker_queue,
            actor_worker_queues,
            report_queue,
            rollout_storage,
            actor_critic,
            rollout_storage.device,
        )
