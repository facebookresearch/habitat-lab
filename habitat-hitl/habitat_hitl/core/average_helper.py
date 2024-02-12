#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import deque


class AverageHelper:
    def __init__(self, window_size, output_rate):
        self.window_size = window_size
        self.output_rate = output_rate
        self.data = deque(maxlen=window_size)
        self.counter = 0

    def add(self, x):
        self.data.append(x)
        self.counter += 1
        if self.counter % self.output_rate == 0:
            return self.get_average()
        else:
            return None

    def get_average(self):
        return sum(self.data) / len(self.data) if self.data else None
