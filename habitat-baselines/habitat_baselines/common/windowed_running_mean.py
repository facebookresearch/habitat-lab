#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import numbers
from typing import Optional, Sequence, Union

import attr
import numpy as np


@attr.s(auto_attribs=True, slots=True, repr=False)
class WindowedRunningMean:
    r"""Efficient implementation of a windowed running mean. Supports an infinite window"""
    window_size: Union[int, float]
    _sum: float = attr.ib(0.0, init=False)
    _count: int = attr.ib(0, init=False)
    _ptr: int = attr.ib(0, init=False)
    _buffer: Optional[np.ndarray] = attr.ib(None, init=False)

    def __attrs_post_init__(self):
        if not self.infinite_window:
            self.window_size = int(self.window_size)
            self._buffer = np.zeros((self.window_size,), dtype=np.float64)

    def add(self, v_i: Union[numbers.Real, float, int]) -> None:
        v = float(v_i)
        self._sum += v
        if self.infinite_window:
            self._count += 1
        else:
            assert isinstance(self.window_size, int)
            if self._count == self.window_size:
                self._sum -= float(self._buffer[self._ptr])
            else:
                self._count += 1

            self._buffer[self._ptr] = v
            self._ptr = (self._ptr + 1) % self.window_size

    def add_many(self, vs: Sequence[Union[numbers.Real, float, int]]):
        for v in vs:
            self.add(v)

    @property
    def mean(self) -> float:
        return self.sum / max(self.count, 1)

    @property
    def sum(self) -> float:
        return self._sum

    @property
    def infinite_window(self) -> bool:
        return math.isinf(self.window_size) or self.window_size <= 0

    @property
    def count(self) -> int:
        return self._count

    def __iadd__(self, v):
        self.add(v)
        return self

    def __float__(self):
        return float(self.mean)

    def __repr__(self):
        return "WindowedRunningMean(window_size={}, mean={})".format(
            self.window_size, self.mean
        )
