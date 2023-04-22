#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

from habitat.utils.perf_logger import PerfLogger
from habitat_baselines.common.windowed_running_mean import WindowedRunningMean

EPS = 1e-5


class TimingContext:
    def __init__(
        self, timer, key, additive=False, average=None, perf_logger=None
    ):
        self._timer = timer
        self._key = key
        self._additive = additive
        self._average = average
        self._time_enter = None
        self._perf_logger: PerfLogger = perf_logger

    def __enter__(self):
        if self._key not in self._timer:
            if self._average is not None:
                self._timer[self._key] = WindowedRunningMean(self._average)
            else:
                self._timer[self._key] = 0

        self._time_enter = time.perf_counter()

    def __exit__(self, type_, value, traceback):
        raw_time_passed = time.perf_counter() - self._time_enter
        time_passed = max(raw_time_passed, EPS)  # EPS to prevent div by zero

        if self._additive or self._average is not None:
            self._timer[self._key] += time_passed
        else:
            self._timer[self._key] = time_passed

        if self._perf_logger:
            self._perf_logger.add_sample_stat(self._key, raw_time_passed)


class Timing(dict):
    # The optional PerfLogger here works with calls to avg_time to help inspect those
    # timings. See also PerfLogger.check_log_summary.
    def __init__(self, perf_logger=None):
        self._perf_logger = perf_logger

    def timeit(self, key):
        return TimingContext(self, key)

    def add_time(self, key):
        return TimingContext(self, key, additive=True)

    def avg_time(self, key, average=float("inf")):
        return TimingContext(
            self, key, average=average, perf_logger=self._perf_logger
        )

    def __str__(self):
        s = ""
        i = 0
        for key, value in self.items():
            str_value = f"{float(value):.4f}"
            s += f"{key}: {str_value}"
            if i < len(self) - 1:
                s += ", "
            i += 1
        return s
