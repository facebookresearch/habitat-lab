#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from contextlib import nullcontext
from functools import wraps

from habitat_baselines.common.windowed_running_mean import WindowedRunningMean

EPS = 1e-5


class TimingContext:
    def __init__(self, timer, key, additive=False, average=None):
        self._timer = timer
        self._key = key
        self._additive = additive
        self._average = average
        self._time_enter = None

    def __call__(self, f):
        @wraps(f)
        def wrapper(*args, **kw):
            with self:
                return f(*args, **kw)

        return wrapper

    def __enter__(self):
        if self._key not in self._timer:
            if self._average is not None:
                self._timer[self._key] = WindowedRunningMean(self._average)
            else:
                self._timer[self._key] = 0

        self._time_enter = time.perf_counter()

    def __exit__(self, type_, value, traceback):
        time_passed = max(
            time.perf_counter() - self._time_enter, EPS
        )  # EPS to prevent div by zero

        if self._additive or self._average is not None:
            self._timer[self._key] += time_passed
        else:
            self._timer[self._key] = time_passed


class EmptyContext(nullcontext):
    def __call__(self, f):
        return f


class Timing(dict):
    def __init__(self, timing_level_threshold: int = 0):
        """
        :param timing_level: The minimum allowed timing log level. The higher
            this value the more that is filtered out. `0` is for the
            highest priority timings. So if `timing_level=1`, all timings
            registered with `timing_level` with 0 or 1 are registered, but a timing
            level of 2 is skipped.
        """
        self._timing_level_threshold = timing_level_threshold

    def timeit(self, key):
        return TimingContext(self, key)

    def add_time(self, key):
        return TimingContext(self, key, additive=True)

    def avg_time(self, key, average=float("inf"), level=0):
        """
        :param level: By default the timing level is 0, and the timing
            will be registered. A higher timing level could be filtered.
        """

        if level > self._timing_level_threshold:
            return EmptyContext()
        return TimingContext(self, key, average=average)

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


# The level of logging is controlled via this environment variable. The higher
# the value, the more is logged. Will log if the `level` value in
# `timer.avg_time` is <= this value.
timing_level = int(os.environ.get("HABITAT_TIMING_LEVEL", 0))

# Global timer instance. All its metrics are reported in `ppo_trainer.py`
g_timer = Timing(timing_level)
