import contextlib
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
    num_steps_done: torch.Tensor
    done_event: Event
    flush_secs: int = 30

    def _world_size(self):
        if not torch.distributed.is_initialized():
            return 1

        return torch.distributed.get_world_size()

    def _reduce(self, arr):
        if not torch.distributed.is_initialized():
            return arr

        t = torch.from_numpy(arr).to(device=self.device, copy=True)
        torch.distributed.all_reduce(t)
        return t.cpu().numpy()

    def _reduce_dict(self, d):
        if not torch.distributed.is_initialized():
            return d

        keys = sorted(d.keys())
        stats = self._reduce(np.array([d[k] for k in keys], dtype=np.float32))

        return {k: type(d[k])(stats[i]) for i, k in enumerate(keys)}

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

    def log_metrics(self, writer: TensorboardWriter, learner_metrics):
        keys = sorted(self.window_episode_stats.keys())
        stats = [
            self.window_episode_stats[k][-1] - self.window_episode_stats[k][0]
            for k in keys
        ]

        learner_keys = sorted(learner_metrics.keys())
        stats += [learner_metrics[k] for k in learner_keys]

        stats += [self.steps_delta, (time.time() - self.start_time)]

        stats = self._reduce(np.array(stats, dtype=np.float32))
        stats[0 : len(keys)] /= (
            len(self.window_episode_stats["reward"]) * self._world_size()
        )
        stats[len(keys) : len(keys) + len(learner_keys)] /= self._world_size()
        self.num_steps_done[:] += int(stats[len(keys) + len(learner_keys)])
        self.time_taken = (
            stats[len(keys) + len(learner_keys) + 1] / self._world_size()
        )

        self.steps_delta = 0

        if rank0_only():
            n_steps = int(self.num_steps_done)
            writer.add_scalar("reward", stats[keys.index("reward")], n_steps)
            metrics = {
                k: stats[i] for i, k in enumerate(keys) if k != "reward"
            }
            if len(metrics) > 0:
                writer.add_scalars("metrics", metrics, n_steps)

            writer.add_scalars(
                "losses",
                {
                    k: stats[len(keys) + i]
                    for i, k in enumerate(learner_keys)
                    if k != "dist_entropy"
                },
                n_steps,
            )

            writer.add_scalars(
                "perf", {"fps": n_steps / self.time_taken}, n_steps
            )

        if self.n_update_reports % self.config.LOG_INTERVAL == 0:
            if rank0_only():
                logger.info(
                    "update: {}\tfps: {:.3f}\tframes: {:d}".format(
                        self.n_update_reports,
                        n_steps / self.time_taken,
                        n_steps,
                    )
                )
                logger.info(
                    "Average window size: {}  {}".format(
                        len(self.window_episode_stats["reward"]),
                        "  ".join(
                            "{}: {:.3f}".format(k, stats[i])
                            for i, k in enumerate(keys)
                        ),
                    )
                )
            for name in sorted(self.timing_stats.keys()):
                stats = self._reduce_dict(self.timing_stats[name])
                if rank0_only():
                    logger.info(
                        "{}: ".format(name)
                        + "  ".join(
                            "{}: {:.1f}ms".format(k, v / stats["count"])
                            for k, v in stats.items()
                            if k != "count"
                        )
                    )

    def run(self):
        if get_distrib_size()[2] > 1:
            local_rank, _ = init_distrib_slurm(backend="nccl", port_offset=1)
            self.device = torch.device("cuda", local_rank)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        ppo_cfg = self.config.RL.PPO

        self.steps_delta = 0
        self.running_episode_stats = defaultdict(lambda: 0.0)
        self.window_episode_stats = defaultdict(
            lambda: deque(
                maxlen=ppo_cfg.reward_window_size
                * self.config.NUM_ENVIRONMENTS
            )
        )
        self.timing_stats = {
            n: defaultdict(lambda: 0.0) for n in ("learner", "policy", "actor")
        }
        self.start_time = time.time()
        self.n_update_reports = 0

        with (
            TensorboardWriter(
                self.config.TENSORBOARD_DIR,
                flush_secs=self.flush_secs,
                purge_step=int(self.num_steps_done),
            )
            if rank0_only()
            else contextlib.suppress()
        ) as writer:
            while not self.done_event.is_set():
                try:
                    datas = self.report_queue.get_many(timeout=1.0)
                except queue.Empty:
                    continue

                for task, data in datas:
                    if task == ReportWorkerTasks.episode_end:
                        self.running_episode_stats["reward"] += data["reward"]
                        for k, v in self._extract_scalars_from_info(
                            data["info"]
                        ):
                            self.running_episode_stats[k] += v

                        for k, v in self.running_episode_stats.items():
                            self.window_episode_stats[k].append(v)

                        self.steps_delta += data["length"]

                    elif task == ReportWorkerTasks.learner_update:
                        self.n_update_reports += 1
                        self.log_metrics(writer, data)
                    elif task == ReportWorkerTasks.learner_timing:
                        for k, v in data.items():
                            self.timing_stats["learner"][k] += v.value()

                        self.timing_stats["learner"]["count"] += 1
                    elif task == ReportWorkerTasks.policy_timing:
                        for k, v in data.items():
                            self.timing_stats["policy"][k] += v.value()

                        self.timing_stats["policy"]["count"] += 1
                    elif task == ReportWorkerTasks.actor_timing:
                        for k, v in data.items():
                            self.timing_stats["actor"][k] += v.value()

                        self.timing_stats["actor"]["count"] += 1
                    else:
                        raise RuntimeError(f"ReportWorker unknown task {task}")


class ReportWorker(WorkerBase):
    def __init__(
        self,
        mp_ctx: BaseContext,
        config: Config,
        report_queue: faster_fifo.Queue,
        init_num_steps=0,
    ):
        self.num_steps_done = torch.full(
            (1,), init_num_steps, dtype=torch.int64
        )
        self.num_steps_done.share_memory_()
        super().__init__(
            mp_ctx,
            ReportWorkerProcess,
            config,
            report_queue,
            self.num_steps_done,
        )
