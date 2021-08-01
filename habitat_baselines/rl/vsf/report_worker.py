import queue
import time
from collections import defaultdict, deque
from multiprocessing import Event
from multiprocessing.context import BaseContext
from typing import Any, Dict

import attr
import faster_fifo
import numpy as np
import torch

from habitat import Config, logger
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ddppo.ddp_utils import (
    get_distrib_size,
    init_distrib_slurm,
    rank0_only,
)
from habitat_baselines.rl.vsf.task_enums import ReportWorkerTasks
from habitat_baselines.rl.vsf.worker_common import ProcessBase, WorkerBase


@attr.s(auto_attribs=True)
class ReportWorkerProcess(ProcessBase):
    config: Config
    report_queue: faster_fifo.Queue
    done_event: Event

    def _reduce(self, arr):
        if not torch.distributed.is_initialized():
            return arr

        t = torch.from_numpy(arr)
        torch.distributed.reduce(t, 0)
        return (t / torch.distributed.get_world_size()).numpy()

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision"}

    @classmethod
    def _extract_scalars_from_info(cls, info: Dict[str, Any]):
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                for subk, subv in cls._extract_scalars_from_info(v):
                    yield f"{k}.{subk}", subv
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                yield k, float(v)

    def log(self):
        keys = sorted(self.window_episode_stats.keys())
        deltas = np.array(
            [
                self.window_episode_stats[k][-1]
                - self.window_episode_stats[k][0]
                for k in keys
            ]
        ) / len(self.window_episode_stats["reward"])

        deltas = self._reduce(deltas)

        if rank0_only():
            logger.info(
                "update: {}\tframes: {}\tfps: {:.3f}".format(
                    self.n_update_reports,
                    self.total_steps,
                    self.total_steps / (time.time() - self.start_time),
                )
            )
            logger.info(
                "Average window size: {}  {}".format(
                    len(self.window_episode_stats["reward"]),
                    "  ".join(
                        "{}: {:.3f}".format(k, deltas[i])
                        for i, k in enumerate(keys)
                    ),
                )
            )

    def run(self):
        if get_distrib_size()[2] > 1:
            init_distrib_slurm(backend="gloo")

        ppo_cfg = self.config.RL.PPO

        self.total_steps = 0.0
        self.running_episode_stats = defaultdict(lambda: 0.0)
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )
        self.start_time = time.time()
        self.n_update_reports = 0
        self._logged = False

        while not self.done_event.is_set():
            if (
                ((self.n_update_reports + 1) % self.config.LOG_INTERVAL) == 0
                and len(self.window_episode_stats["reward"]) > 2
                and not self._logged
            ):
                self.log()
                self._logged = True

            try:
                datas = self.report_queue.get_many(timeout=1.0)
            except queue.Empty:
                continue

            for task, data in datas:
                if task == ReportWorkerTasks.episode_end:
                    self.running_episode_stats["reward"] += data["reward"]
                    for k, v in self._extract_scalars_from_info(data["info"]):
                        self.running_episode_stats[k] += v

                    for k, v in self.running_episode_stats.items():
                        self.window_episode_stats[k].append(v)

                    self.total_steps += data["length"]

                elif task == ReportWorkerTasks.learner_update:
                    #  for k, v in self._extract_scalars_from_info(data):
                    #  pass
                    self.n_update_reports += 1
                    self._logged = False
                else:
                    continue
                    raise RuntimeError(f"ReportWorker unknown task {task}")


class ReportWorker(WorkerBase):
    def __init__(
        self,
        mp_ctx: BaseContext,
        config: Config,
        report_queue: faster_fifo.Queue,
    ):
        super().__init__(mp_ctx, ReportWorkerProcess, config, report_queue)
