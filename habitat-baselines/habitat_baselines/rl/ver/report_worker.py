#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import functools
import os
import time
from collections import defaultdict
from multiprocessing.context import BaseContext
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

import attr
import numpy as np
import torch

from habitat import logger
from habitat_baselines.common.tensor_dict import (
    NDArrayDict,
    transpose_list_of_dicts,
)
from habitat_baselines.common.tensorboard_utils import (
    TensorboardWriter,
    get_writer,
)
from habitat_baselines.common.windowed_running_mean import WindowedRunningMean
from habitat_baselines.rl.ddppo.ddp_utils import (
    gather_objects,
    get_distrib_size,
    init_distrib_slurm,
    rank0_only,
)
from habitat_baselines.rl.ver.queue import BatchedQueue
from habitat_baselines.rl.ver.task_enums import ReportWorkerTasks
from habitat_baselines.rl.ver.worker_common import ProcessBase, WorkerBase
from habitat_baselines.utils.info_dict import extract_scalars_from_info

if TYPE_CHECKING:
    from omegaconf import DictConfig


@attr.s(auto_attribs=True)
class ReportWorkerProcess(ProcessBase):
    r"""Responsible for generating reports. Reports on system performance (timings),
    learning progress, and agent training progress.
    """
    port: int
    config: "DictConfig"
    report_queue: BatchedQueue
    my_t_zero: float
    num_steps_done: torch.Tensor
    time_taken: torch.Tensor
    n_update_reports: int = 0
    flush_secs: int = 30
    _world_size: Optional[int] = None
    _prev_time_taken: float = 0.0
    timing_stats: Dict[str, Dict[str, WindowedRunningMean]] = attr.ib(
        init=False, default=None
    )
    stats_this_rollout: Dict[str, List[float]] = attr.ib(
        init=False, factory=lambda: defaultdict(list)
    )
    steps_delta: int = 0
    writer: Optional[Any] = None
    run_id: Optional[str] = None
    preemption_decider_report: Dict[str, float] = attr.ib(
        factory=dict, init=False
    )
    window_episode_stats: Optional[Dict[str, WindowedRunningMean]] = attr.ib(
        default=None, init=False
    )

    def __attrs_post_init__(self):
        self.build_dispatch_table(ReportWorkerTasks)

    def state_dict(self):
        self.response_queue.put(
            dict(
                prev_time_taken=float(self.time_taken),
                window_episode_stats=self.window_episode_stats,
                num_steps_done=int(self.num_steps_done),
                timing_stats=self.timing_stats,
                running_frames_window=self.running_frames_window,
                running_time_window=self.running_time_window,
                n_update_reports=self.n_update_reports,
                run_id=self.writer.get_run_id()
                if self.writer is not None
                else None,
            )
        )

    def load_state_dict(self, state_dict):
        self._prev_time_taken = state_dict["prev_time_taken"]
        self.time_taken.fill_(self._prev_time_taken)
        self.window_episode_stats = state_dict["window_episode_stats"]
        self.num_steps_done.fill_(int(state_dict["num_steps_done"]))
        self.timing_stats = state_dict["timing_stats"]
        self.running_frames_window = state_dict["running_frames_window"]
        self.running_time_window = state_dict["running_time_window"]
        self.n_update_reports = state_dict["n_update_reports"]

    @property
    def world_size(self) -> int:
        if self._world_size is None:
            if not torch.distributed.is_initialized():
                self._world_size = 1
            else:
                self._world_size = torch.distributed.get_world_size()

        return self._world_size

    def _all_reduce(self, val, reduce_op=torch.distributed.ReduceOp.SUM):
        if self.world_size == 1:
            return val

        t = torch.as_tensor(val, dtype=torch.float64)
        torch.distributed.all_reduce(t, op=reduce_op)
        return type(val)(t)

    def get_time(self):
        return time.perf_counter() - self.my_t_zero

    def log_metrics(
        self, writer: TensorboardWriter, learner_metrics: Dict[str, float]
    ):
        self.steps_delta = int(self._all_reduce(self.steps_delta))
        self.num_steps_done += self.steps_delta
        last_time_taken = float(self.time_taken)
        self.time_taken.fill_(
            self._all_reduce(self.get_time() - self.start_time)
            / self.world_size
            + self._prev_time_taken
        )

        self.running_frames_window += self.steps_delta
        self.running_time_window += float(self.time_taken) - last_time_taken

        self.steps_delta = 0

        all_stats_this_rollout = gather_objects(
            dict(self.stats_this_rollout), device=self.device
        )
        self.stats_this_rollout.clear()
        if rank0_only():
            assert all_stats_this_rollout is not None
            assert self.window_episode_stats is not None
            for stats in all_stats_this_rollout:
                for k, vs in stats.items():
                    self.window_episode_stats[k].add_many(vs)

        all_learner_metrics = gather_objects(learner_metrics)
        all_preemption_decider_reports = gather_objects(
            self.preemption_decider_report
        )

        if rank0_only():
            assert all_learner_metrics is not None
            learner_metrics = cast(
                Dict[str, float],
                (
                    NDArrayDict.from_tree(
                        transpose_list_of_dicts(*all_learner_metrics)
                    )
                    .map(np.mean)
                    .to_tree()
                ),
            )
            n_steps = int(self.num_steps_done)
            if "reward" in self.window_episode_stats:
                writer.add_scalar(
                    "reward",
                    self.window_episode_stats["reward"].mean,
                    n_steps,
                )

            for k in self.window_episode_stats.keys():
                if k == "reward":
                    continue

                writer.add_scalar(
                    f"metrics/{k}", self.window_episode_stats[k].mean, n_steps
                )

            for k, v in learner_metrics.items():
                writer.add_scalar(f"learner/{k}", v, n_steps)

            writer.add_scalar(
                "perf/fps",
                n_steps / float(self.time_taken),
                n_steps,
            )
            writer.add_scalar(
                "perf/fps_window",
                float(self.running_frames_window)
                / float(self.running_time_window),
                n_steps,
            )

        if rank0_only():
            assert all_preemption_decider_reports is not None
            preemption_decider_report = cast(
                Dict[str, float],
                NDArrayDict.from_tree(
                    transpose_list_of_dicts(*all_preemption_decider_reports)
                )
                .map(np.mean)
                .to_tree(),
            )
            preemption_decider_report = {
                k: v / (self.world_size if "time" in k else 1)
                for k, v in preemption_decider_report.items()
            }
            for k, v in preemption_decider_report.items():
                writer.add_scalar(f"preemption_decider/{k}", v, n_steps)

        if (
            self.n_update_reports % self.config.habitat_baselines.log_interval
            == 0
        ):
            if rank0_only():
                logger.info(
                    "update: {}\tfps: {:.1f}\twindow fps: {:.1f}\tframes: {:d}".format(
                        self.n_update_reports,
                        n_steps / float(self.time_taken),
                        float(self.running_frames_window)
                        / float(self.running_time_window),
                        n_steps,
                    )
                )
                if len(self.window_episode_stats) > 0:
                    logger.info(
                        "Average window size: {}  {}".format(
                            next(
                                iter(self.window_episode_stats.values())
                            ).count,
                            "  ".join(
                                "{}: {:.3f}".format(k, v.mean)
                                for k, v in self.window_episode_stats.items()
                            ),
                        )
                    )
            all_timing_stats = gather_objects(
                {
                    k: {sk: sv.mean for sk, sv in v.items()}
                    for k, v in self.timing_stats.items()
                }
            )
            if rank0_only():
                assert all_timing_stats is not None
                timing_stats = cast(
                    Dict[str, Dict[str, float]],
                    NDArrayDict.from_tree(
                        transpose_list_of_dicts(*all_timing_stats)
                    )
                    .map(np.mean)
                    .to_tree(),
                )
                for stats_name in sorted(timing_stats.keys()):
                    logger.info(
                        "{}: ".format(stats_name)
                        + "  ".join(
                            "{}: {:.1f}ms".format(k, v * 1e3)
                            for k, v in sorted(
                                timing_stats[stats_name].items(),
                                key=lambda kv: kv[1],
                                reverse=True,
                            )
                        )
                    )

    def episode_end(self, data):
        self.stats_this_rollout["reward"].append(data["reward"])
        for k, v in extract_scalars_from_info(data["info"]).items():
            self.stats_this_rollout[k].append(v)

    def num_steps_collected(self, num_steps: int):
        self.steps_delta = num_steps

    def learner_update(self, data):
        self.n_update_reports += 1
        self.log_metrics(self.writer, data)

    def start_collection(self, start_time):
        start_time = self._all_reduce(
            start_time - self.my_t_zero,
            reduce_op=torch.distributed.ReduceOp.MIN,
        )
        self.start_time = start_time

    def preemption_decider(self, preemption_decider_report):
        self.preemption_decider_report = preemption_decider_report

    def env_timing(self, timing):
        for k, v in timing.items():
            self.timing_stats["env"][k] += v

    def policy_timing(self, timing):
        for k, v in timing.items():
            self.timing_stats["policy"][k] += v

    def learner_timing(self, timing):
        for k, v in timing.items():
            self.timing_stats["learner"][k] += v

    def get_window_episode_stats(self):
        self.response_queue.put(self.window_episode_stats)

    @property
    def task_queue(self) -> BatchedQueue:
        return self.report_queue

    def run(self):
        self.device = torch.device("cpu")
        if get_distrib_size()[2] > 1:
            os.environ["MAIN_PORT"] = str(self.port)
            init_distrib_slurm(backend="gloo")
            torch.distributed.barrier()

        self.response_queue.put(None)

        ppo_cfg = self.config.habitat_baselines.rl.ppo

        self.steps_delta = 0
        if rank0_only():
            self.window_episode_stats = defaultdict(
                functools.partial(
                    WindowedRunningMean,
                    ppo_cfg.reward_window_size
                    * self.config.habitat_baselines.num_environments
                    * self.world_size,
                )
            )
        else:
            self.window_episode_stats = None

        timing_types = {
            ReportWorkerTasks.env_timing: "env",
            ReportWorkerTasks.policy_timing: "policy",
            ReportWorkerTasks.learner_timing: "learner",
        }
        self.timing_stats = {
            n: defaultdict(
                functools.partial(
                    WindowedRunningMean, ppo_cfg.reward_window_size
                )
            )
            for n in timing_types.values()
        }
        self.preemption_decider_report = {}

        self.start_time = self.get_time()
        self.running_time_window = WindowedRunningMean(
            ppo_cfg.reward_window_size
        )
        self.running_frames_window = WindowedRunningMean(
            ppo_cfg.reward_window_size
        )

        with (
            get_writer(
                self.config,
                resume_run_id=self.run_id,
                flush_secs=self.flush_secs,
                purge_step=int(self.num_steps_done),
            )
            if rank0_only()
            else contextlib.suppress()
        ) as writer:
            self.writer = writer
            super().run()

            self.writer = None


class ReportWorker(WorkerBase):
    def __init__(
        self,
        mp_ctx: BaseContext,
        port: int,
        config: "DictConfig",
        report_queue: BatchedQueue,
        my_t_zero: float,
        init_num_steps=0,
        run_id=None,
    ):
        self.num_steps_done = torch.full(
            (), int(init_num_steps), dtype=torch.int64
        )
        self.time_taken = torch.full((), 0.0, dtype=torch.float64)
        self.num_steps_done.share_memory_()
        self.time_taken.share_memory_()
        self.report_queue = report_queue
        super().__init__(
            mp_ctx,
            ReportWorkerProcess,
            port,
            config,
            report_queue,
            my_t_zero,
            self.num_steps_done,
            self.time_taken,
            run_id=run_id,
        )

        self.response_queue.get()

    def start_collection(self):
        self.report_queue.put(
            (ReportWorkerTasks.start_collection, time.perf_counter())
        )

    def state_dict(self):
        self.report_queue.put((ReportWorkerTasks.state_dict, None))
        return self.response_queue.get()

    def load_state_dict(self, state_dict):
        if state_dict is not None:
            self.report_queue.put(
                (ReportWorkerTasks.load_state_dict, state_dict)
            )

    def get_window_episode_stats(self):
        self.report_queue.put(
            (ReportWorkerTasks.get_window_episode_stats, None)
        )
        return self.response_queue.get()
