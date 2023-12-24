#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time


class AverageRateTracker:
    def __init__(self, duration_window: float) -> None:
        self._recent_count: int = 0
        self._recent_time: float = time.time()
        self._duration_window: float = duration_window
        self._recent_rate: float = 0.0

    def increment(self, inc: int) -> None:
        self._recent_count += inc

    def get_smoothed_rate(self) -> float:
        current_time = time.time()
        elapsed_time = current_time - self._recent_time
        if elapsed_time > self._duration_window:
            self._recent_rate = self._recent_count / elapsed_time
            self._recent_count = 0
            self._recent_time = current_time

        return self._recent_rate
