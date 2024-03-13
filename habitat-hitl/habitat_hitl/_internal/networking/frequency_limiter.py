#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import time


class FrequencyLimiter:
    def __init__(self, desired_frequency: float) -> None:
        self.desired_frequency: float = desired_frequency
        self.desired_period: float = (
            1.0 / desired_frequency if desired_frequency else None
        )
        self.timer: float = time.time()

    def _calculate_remaining_time(self):
        if not self.desired_period:
            return None

        current_time = time.time()
        elapsed_time = current_time - self.timer

        # Don't let timer get too far behind
        if elapsed_time > self.desired_period * 2.0:
            self.timer = current_time - self.desired_period * 2.0
            elapsed_time = current_time - self.timer

        # Calculate the remaining time to meet the desired period
        remaining_time = self.desired_period - elapsed_time
        return remaining_time

    def limit_frequency(self):
        remaining_time = self._calculate_remaining_time()
        if remaining_time is not None and remaining_time > 0:
            time.sleep(remaining_time)
        self.timer += self.desired_period

    async def limit_frequency_async(self):
        remaining_time = self._calculate_remaining_time()
        if remaining_time is None:
            return
        if remaining_time is not None and remaining_time > 0:
            await asyncio.sleep(remaining_time)
        self.timer += self.desired_period
