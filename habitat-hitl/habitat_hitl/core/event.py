#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, List


class Event:
    def __init__(self):
        self._callbacks: List[Callable[[Any], None]] = []

    def registerCallback(self, callback):
        self._callbacks.append(callback)

    def invoke(self, obj: Any):
        for callback in self._callbacks:
            callback(obj)
