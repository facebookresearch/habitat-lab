#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import random
import subprocess
import time
from collections import OrderedDict

import torch

from habitat import logger
from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.tasks.nav.nav import ImageGoalSensor


class PerfLogger:
    """
    The PerfLogger logs runtime perf stats as name/value pairs on two rows, like so:

    2023-04-19 19:17:41,331 PerfLogger
    statName1,statName2,statName3
    1.23,4,anotherStatValue

    The comma-separated lines can be pasted into a spreadsheet as CSV data. Runtime
    perf stats include timings and measures of scene complexity like object count and
    triangle count. These can be compared across experiments to understand changes
    in training throughput (steps per second, aka SPS).

    To enable in PPO training:
    * set config.habitat_baselines.profiling.enable_perf_logger
    * add "runtime_perf_stats" to config.habitat.task.measurements
    """

    def _find_username(self, config):
        if len(config.habitat_baselines.wb.entity):
            return config.habitat_baselines.wb.entity
        return os.getlogin()

    def _find_git_hash(self):
        # from https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
        # Note that link suggests `import git` is not a good idea in long-running
        # processes (like RL training), so we avoid it.

        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("ascii")
            .strip()
        )

    def _check_rank_and_find_num_workers(self):
        if not torch.distributed.is_initialized():
            return 1
        else:
            if torch.distributed.get_rank() != 0:
                logger.warn(
                    "PerfLogger is running for worker rank=="
                    + torch.distributed.get_rank()
                    + " but we recommend only running it on worker rank==0"
                )
            return torch.distributed.get_world_size()

    def _find_job_id(self):
        job_id = os.environ.get("SLURM_JOB_ID", None)
        if job_id is None:
            # use a random int that will generally be unique across train runs
            random.seed()
            job_id = str(random.randint(10000, 99999))
        return job_id

    def __init__(self, config, num_scenes, observation_space, job_id=None):
        if "runtime_perf_stats" not in config.habitat.task.measurements:
            logger.warning(
                "You have profiling.enable_perf_logger==True but you didn't include runtime_perf_stats in habitat.task.measurements. You won't get the full set of runtime perf stats."
            )

        self._log_counter = 0
        self._log_skip_interval = (
            config.habitat_baselines.profiling.perf_logger_skip_interval
        )
        self._num_steps = 0
        self._step_timer = None
        self._stats = OrderedDict()

        self._stats["user"] = self._find_username(config)
        self._stats["job id"] = (
            job_id if job_id is not None else self._find_job_id()
        )
        # self._stats["num steps"] = 0
        self._stats["SPS"] = 0
        self._stats["git hash"] = self._find_git_hash()
        self._stats["GPU"] = (
            torch.cuda.get_device_name()
            if torch.cuda.is_available()
            else "none"
        )
        self._stats["num workers"] = self._check_rank_and_find_num_workers()
        self._stats["num envs"] = config.habitat_baselines.num_environments
        self._stats["num scenes"] = num_scenes
        self._stats["backbone"] = config.habitat_baselines.rl.ddppo.backbone

        num_rgb = 0
        num_depth = 0
        self._stats["viz_types"] = ""
        viz_shapes_set = set()
        for k, v in observation_space.spaces.items():
            if len(v.shape) > 1 and k != ImageGoalSensor.cls_uuid:
                image_shape = (v.shape[0], v.shape[1])
                if image_shape not in viz_shapes_set:
                    viz_shapes_set.add(image_shape)

                if v.shape[2] == 1:
                    num_depth += 1
                elif v.shape[2] == 3:
                    num_rgb += 1
                else:
                    logger.warn(
                        "Failed to understand observation space "
                        + k
                        + " with shape "
                        + v.shape
                    )
        # example outputs for below: Depth, RGB, RGBD, RGB 2x, RGB 2x + Depth
        # todo: figure out how to detect semantic
        if num_depth == 0 and num_rgb == 0:
            self._stats["viz_types"] = "Blind"
        elif num_rgb == 0:
            self._stats["viz_types"] = "Depth"
            if num_depth > 1:
                self._stats["viz_types"] += " " + str(num_depth) + "x"
        elif num_depth == 0:
            self._stats["viz_types"] = "RGB"
            if num_rgb > 1:
                self._stats["viz_types"] += " " + str(num_rgb) + "x"
        elif num_depth == num_rgb:
            self._stats["viz_types"] = "RGBD"
            if num_depth > 1:
                self._stats["viz_types"] += " " + str(num_depth) + "x"
        else:
            self._stats["viz_types"] = "RGB"
            if num_rgb > 1:
                self._stats["viz_types"] += " " + str(num_rgb) + "x"
            self._stats["viz_types"] += "+ Depth"
            if num_depth > 1:
                self._stats["viz_types"] += " " + str(num_depth) + "x"

        self._stats["viz_shapes"] = ";".join(
            [str(shape[0]) + "x" + str(shape[1]) for shape in viz_shapes_set]
        )

        # todo: get top-level config name

        self._reset_sample_stats()

    def _reset_sample_stats(self):
        self._sample_stat_count_average_max = {}

    def _stat_val_to_pretty_string(self, val):
        if isinstance(val, float):
            return "{:0.4}".format(val)
        else:
            return str(val)

    def add_steps(self, num_steps):
        if self._step_timer is None:
            self._step_timer = time.time()
            self._num_steps = 0
        else:
            self._num_steps += num_steps

    def add_sample_stats(self, sample_stat_vals_dict):
        for name, sample_val in sample_stat_vals_dict.items():
            self.add_sample_stat(name, sample_val)

    def add_sample_stat(self, name, sample_val):
        if name not in self._sample_stat_count_average_max:
            self._sample_stat_count_average_max[name] = [0, 0, sample_val]
        stat_list = self._sample_stat_count_average_max[name]
        old_count = stat_list[0]
        stat_list[1] = (stat_list[1] * old_count + sample_val) / (
            old_count + 1
        )
        stat_list[2] = max(stat_list[2], sample_val)
        stat_list[0] = old_count + 1

    def check_log_summary(self, reset_sample_stats=True):
        self._log_counter += 1
        if self._log_counter % self._log_skip_interval != 1:
            return

        if self._log_counter != 1:
            # update SPS stat
            t_curr = time.time()
            elapsed_since_last_log = t_curr - self._step_timer
            # self._stats["num steps"] = self._num_steps
            self._stats["SPS"] = self._num_steps / elapsed_since_last_log
            self._num_steps = 0
            self._step_timer = t_curr

            # non-sample-based stats first, then sample count info, then sample-based stats
            # date/time (comes from log), username, git hash, config name, num envs, GPU, total num updates, updates per summary, **avg_times, max_times, **avg_env_stats, max_env_stats

            all_stat_vals = list(self._stats.values())
            all_stat_names = list(self._stats.keys())
            for name, stat_list in self._sample_stat_count_average_max.items():
                all_stat_names.extend([name + " avg", "max", "count"])
                all_stat_vals.extend(
                    [stat_list[1], stat_list[2], stat_list[0]]
                )

            all_stat_pretty_vals = [
                self._stat_val_to_pretty_string(val) for val in all_stat_vals
            ]

            s = "PerfLogger\n"
            s += ",".join(all_stat_names) + "\n"
            s += ",".join(all_stat_pretty_vals)
            logger.info(s)
        else:
            # first batch of stats are usually not representative, so don't print them
            pass

        if reset_sample_stats:
            self._reset_sample_stats()


@registry.register_measure
class RuntimePerfStats(Measure):
    cls_uuid: str = "runtime_perf_stats"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return RuntimePerfStats.cls_uuid

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        super().__init__()

    def reset_metric(self, *args, **kwargs):
        self._metric = {}

    def update_metric(self, *args, task, **kwargs):
        self._metric = self._sim.get_runtime_perf_stats()
