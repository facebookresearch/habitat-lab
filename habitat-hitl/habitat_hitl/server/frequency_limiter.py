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
        self.last_execution_time: float = time.time()

    def limit_frequency(self) -> None:
        if not self.desired_period:
            return

        current_time = time.time()
        elapsed_time = current_time - self.last_execution_time

        # Calculate the remaining time to meet the desired period
        remaining_time = self.desired_period - elapsed_time

        # Delay the loop execution if needed
        if remaining_time > 0:
            time.sleep(remaining_time)

        self.last_execution_time = time.time()

    async def limit_frequency_async(self) -> None:
        if not self.desired_period:
            return

        current_time = time.time()
        elapsed_time = current_time - self.last_execution_time
        remaining_time = self.desired_period - elapsed_time

        # Delay the loop execution if needed
        if remaining_time > 0:
            await asyncio.sleep(remaining_time)

        self.last_execution_time = time.time()
