#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from typing import Optional


class AverageRateTracker:
    def __init__(self, duration_window: float) -> None:
        self._recent_count = 0
        self._recent_time = time.time()
        self._duration_window = duration_window
        self._recent_rate: Optional[float] = None

    def increment(self, inc: int = 1) -> Optional[float]:
        """
        Increments count for the purpose of tracking rate over time.

        If a new smoothed rate was calculated, it is returned. This is a convenience in case the user wants to regularly log the smoothed framerate. See also get_smoothed_rate.
        """
        self._recent_count += inc
        current_time = time.time()
        elapsed_time = current_time - self._recent_time
        if elapsed_time > self._duration_window:
            self._recent_rate = self._recent_count / elapsed_time
            self._recent_count = 0
            self._recent_time = current_time
            return self._recent_rate
        else:
            return None

    def get_smoothed_rate(self) -> float:
        return self._recent_rate
