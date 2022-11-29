#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import queue
import time
import warnings
from multiprocessing.context import BaseContext
from typing import TYPE_CHECKING

import attr
import numpy as np
import torch

from habitat import logger
from habitat_baselines.common.windowed_running_mean import WindowedRunningMean
from habitat_baselines.rl.ddppo.ddp_utils import init_distrib_slurm, rank0_only
from habitat_baselines.rl.ver.task_enums import (
    PreemptionDeciderTasks,
    ReportWorkerTasks,
)
from habitat_baselines.rl.ver.worker_common import (
    ProcessBase,
    RolloutEarlyEnds,
    WorkerBase,
    WorkerQueues,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


@attr.s(auto_attribs=True)
class PreemptionDeciderProcess(ProcessBase):
    r"""Used to decide when to preempt GPU-worker stragglers when training with VER+DD-PPO.


    This attempts to approximate the optimal preemption schedule for each rollout. The optimal
    preemption time is defined as follows: Given the learner time, LT, and a function Time(S) that returns how long it takes to collect S steps of experience, the number of steps to collect
    is defined as
        argmax_S S / (Time(S) + LT)

    Approximating LT is easy. That's just a simple windowed running mean of learning time as
    that rarely changes between iterations.

    Approximating Time(S) is much harder. To approximate that, we use the average step time
    per environment (per worker) during the last rollout. This is tracked by the Inference Worker
    sending a report to the PreemptionDecider after it has collected a step of experience and written
    it into the learning buffer.
    """
    hostname: str
    port: int
    world_rank: int
    world_size: int
    config: "DictConfig"
    queues: WorkerQueues
    my_t_zero: float
    rollout_ends: RolloutEarlyEnds
    opt_rollout_time_avg: WindowedRunningMean = attr.Factory(
        lambda: WindowedRunningMean(1)
    )
    preemption_error_time_avg: WindowedRunningMean = attr.Factory(
        lambda: WindowedRunningMean(16)
    )
    learner_time_avg: WindowedRunningMean = attr.Factory(
        lambda: WindowedRunningMean(5)
    )
    my_opt_rollout_steps: float = 0.0
    start_time: float = 0.0
    expected_steps_collected: int = 0
    _bin_size: float = 5.0e-3
    _ver_extra_steps_scaling: float = 1.0
    real_steps_collected: int = 0
    n_rollouts_started: int = 0

    def __attrs_post_init__(self):
        self.build_dispatch_table(PreemptionDeciderTasks)

    def _gather(self, arr):
        if self.world_size == 1:
            return arr[np.newaxis]

        if self.world_rank == 0:
            all_arr = np.empty((self.world_size, *arr.shape), dtype=arr.dtype)
        else:
            all_arr = None

        torch.distributed.gather(
            torch.from_numpy(arr),
            gather_list=list(torch.from_numpy(all_arr).unbind(0))
            if self.world_rank == 0
            else None,
            dst=0,
        )

        return all_arr

    def _gather_all_step_averages(self):
        my_averages = np.array(
            [float(v) for v in self.step_averages], dtype=np.float64
        )
        return self._gather(my_averages), my_averages

    def _all_reduce(self, val, reduce_op=torch.distributed.ReduceOp.SUM):
        if self.world_size == 1:
            return val

        t = torch.as_tensor(val, dtype=torch.float64)
        torch.distributed.all_reduce(t, op=reduce_op)
        return type(val)(t)

    def _reduce(self, val, mean: bool = False):
        if self.world_size == 1:
            return val

        t = torch.as_tensor(val, dtype=torch.float64)
        torch.distributed.reduce(t, dst=0)
        if mean:
            t.div_(self.world_size)

        return type(val)(t)

    def _bcast(self, val):
        if self.world_size == 1:
            return val

        t = torch.as_tensor(val)
        torch.distributed.broadcast(t, 0)

        return type(val)(t)

    def _compute_time(
        self,
        all_num_next_steps: np.ndarray,
        all_step_averages: np.ndarray,
        lt: float,
    ):
        # The maximum number of steps we can collect from 1 env. This
        # is rollout length + 1 times the maximum difference in
        # env speed. We also have an additional scaling factor
        max_possible_steps = (
            self.config.habitat_baselines.rl.ppo.num_steps + 1
        ) * (
            self._ver_extra_steps_scaling
            * np.max(all_step_averages)
            / np.min(all_step_averages)
        )

        # A (W, N, T) sized array where each entry represents the time needed
        # to collected t steps of experience from env n on worker w
        rollout_lengths = all_step_averages[..., np.newaxis] * np.arange(
            1, max_possible_steps + 1, dtype=np.float64
        )

        # Bin the rollout lengths and then compute all unique ones.
        # Candidates lengths is in time
        candidate_lengths, candidate_length_counts = np.unique(
            (np.ceil(rollout_lengths / self._bin_size) * self._bin_size),
            return_counts=True,
        )
        # This is the number of steps collected for each candidate length
        candidate_length_steps = np.cumsum(candidate_length_counts)

        # Mask out options where we collect too many steps
        valids = candidate_length_steps <= (
            self.config.habitat_baselines.rl.ppo.num_steps
            * self.config.habitat_baselines.num_environments
            * self.world_size
        )

        candidate_lengths = candidate_lengths[valids]
        candidate_length_steps = candidate_length_steps[valids]

        # Mask out options where we collect too many steps
        # for each worker
        valids = np.all(
            np.count_nonzero(
                rollout_lengths[..., np.newaxis] <= candidate_lengths,
                axis=(1, 2),
            )
            <= all_num_next_steps,
            0,
        )

        candidate_lengths = candidate_lengths[valids]
        candidate_length_steps = candidate_length_steps[valids]

        if not self.config.habitat_baselines.rl.ver.overlap_rollouts_and_learn:
            # The total rollout time is time to collect + learning
            # + some error factor
            total_time = (
                candidate_lengths
                + lt
                + max(float(self.preemption_error_time_avg), 0.0)
            )
        else:
            # If we are overlapping, it's the
            # max between collection time and learning time.
            total_time = candidate_lengths + max(
                float(self.preemption_error_time_avg), 0.0
            )
            if lt > np.max(total_time):
                total_time = lt

        candidate_sps = candidate_length_steps / total_time

        max_sps_idx = np.argmax(candidate_sps)
        target_length_time = candidate_lengths[max_sps_idx]

        if np.any(rollout_lengths[..., -1] <= target_length_time):
            logger.info("Growing scaling factor")
            self._ver_extra_steps_scaling *= 1.2

        self.expected_steps_collected = candidate_length_steps[max_sps_idx]
        return target_length_time, max_possible_steps

    def update(self, num_next_steps: int):
        all_num_next_steps = self._gather(
            np.array([num_next_steps], dtype=np.int64)
        )
        all_step_averages, my_step_averages = self._gather_all_step_averages()
        lt = max(self._reduce(float(self.learner_time_avg), mean=True), 0.01)
        target_length_time = -1.0
        max_possible_steps = 0.0

        if rank0_only():
            if np.all(all_step_averages > 0):
                target_length_time, max_possible_steps = self._compute_time(
                    all_num_next_steps, all_step_averages, lt
                )
            else:
                warnings.warn(
                    "Not all environments have an estimated step time. If this warning is seen only a few times, all is fine."
                )

        target_length_time, max_possible_steps = self._bcast(
            [float(target_length_time), float(max_possible_steps)]
        )

        if target_length_time > 0.0:
            self.opt_rollout_time_avg += target_length_time
            step_times = my_step_averages[:, np.newaxis] * np.arange(
                1, max_possible_steps + 1, dtype=np.float64
            )

            self.my_opt_rollout_steps = np.count_nonzero(
                step_times <= float(self.opt_rollout_time_avg)
            )

        if (
            self.learner_time_avg.count == self.learner_time_avg.window_size
            and self.opt_rollout_time_avg.count
            == self.opt_rollout_time_avg.window_size
        ):
            self.rollout_ends.steps.value = float(self.my_opt_rollout_steps)
        else:
            self.rollout_ends.steps.value = -1

    def policy_step(self, data):
        step_time = data["t_stamp"]
        for _, env_idx in data["steps_finished"]:
            if self.last_step_times[env_idx] > 0:
                self.step_averages[env_idx] += (
                    step_time - self.last_step_times[env_idx]
                )

            self.last_step_times[env_idx] = step_time

        self.real_steps_collected += len(data["steps_finished"])

    def start_rollout(self, start_time: float):
        self.start_time = self._all_reduce(
            (start_time - self.my_t_zero),
            reduce_op=torch.distributed.ReduceOp.MIN,
        )

        self.n_rollouts_started += 1
        self.last_step_times[:] = -1.0

        self.started = True
        if (
            self.learner_time_avg.count == self.learner_time_avg.window_size
            and self.opt_rollout_time_avg.count
            == self.opt_rollout_time_avg.window_size
        ):
            self.rollout_ends.time.value = (
                self.my_t_zero
                + self.start_time
                + float(self.opt_rollout_time_avg)
            )
        else:
            self.rollout_ends.time.value = -1

    def end_rollout(self, end_steps_time: float, num_next_steps: int):
        end_steps_time = self._all_reduce(
            (end_steps_time - self.my_t_zero),
            reduce_op=torch.distributed.ReduceOp.MAX,
        )
        self.queues.report.put(
            (
                ReportWorkerTasks.preemption_decider,
                dict(
                    real_steps_collected=self.real_steps_collected,
                    expected_steps_collected=self.expected_steps_collected,
                    real_rollout_time=(end_steps_time - self.start_time) * 1e3,
                    expected_rollout_time=float(self.opt_rollout_time_avg)
                    * 1e3,
                ),
            )
        )

        if self.rollout_ends.time.value > 0:
            self.preemption_error_time_avg += (
                end_steps_time - self.start_time
            ) - float(self.opt_rollout_time_avg)

        self.started = False
        self.rollout_ends.time.value = -1.0
        self.rollout_ends.steps.value = -1.0

        self.update(num_next_steps)
        self.real_steps_collected = 0

    def learner_time(self, learner_time: float):
        self.learner_time_avg += learner_time

    def run(self):
        if self.world_size > 1:
            os.environ["MAIN_PORT"] = str(self.port)
            init_distrib_slurm(backend="gloo")
            torch.distributed.barrier()

        self.response_queue.put(None)

        self.step_averages = [
            WindowedRunningMean(
                5 * self.config.habitat_baselines.rl.ppo.num_steps
            )
            for _ in range(self.config.habitat_baselines.num_environments)
        ]
        self.last_step_times = np.zeros(
            (self.config.habitat_baselines.num_environments,), dtype=np.float64
        )

        self.started = False
        tasks = []
        self.real_steps_collected = 0
        self.n_rollouts_started = 0
        while not self.done_event.is_set():
            try:
                new_tasks = self.queues.preemption_decider.get_many(
                    timeout=1.0
                )
            except queue.Empty:
                continue

            if not self.started:
                new_tasks.sort(
                    key=lambda t: 0
                    if t[0] == PreemptionDeciderTasks.start_rollout
                    else 1
                )

                if new_tasks[0][0] != PreemptionDeciderTasks.start_rollout:
                    tasks.extend(new_tasks)
                    continue

            tasks = new_tasks + tasks

            did_break = False
            for i, (task, data) in enumerate(tasks):
                self.dispatch_task(task, data)

                if task == PreemptionDeciderTasks.end_rollout:
                    tasks = tasks[i + 1 :]
                    did_break = True
                    break

            if not did_break:
                tasks = []


class PreemptionDeciderWorker(WorkerBase):
    def __init__(
        self,
        mp_ctx: BaseContext,
        hostname: str,
        port: int,
        world_rank: int,
        world_size: int,
        config: "DictConfig",
        queues: WorkerQueues,
        my_t_zero: float,
    ):
        self.rollout_ends = RolloutEarlyEnds(mp_ctx)
        self.queues = queues
        super().__init__(
            mp_ctx,
            PreemptionDeciderProcess,
            hostname,
            port,
            world_rank,
            world_size,
            config,
            queues,
            my_t_zero,
            rollout_ends=self.rollout_ends,
        )

        self.response_queue.get()

    def start_rollout(self):
        self.queues.preemption_decider.put(
            (PreemptionDeciderTasks.start_rollout, time.perf_counter())
        )

    def end_rollout(self, num_next_steps):
        self.queues.preemption_decider.put(
            (
                PreemptionDeciderTasks.end_rollout,
                (time.perf_counter(), num_next_steps),
            )
        )

    def learner_time(self, learning_time):
        self.queues.preemption_decider.put(
            (PreemptionDeciderTasks.learner_time, learning_time)
        )
