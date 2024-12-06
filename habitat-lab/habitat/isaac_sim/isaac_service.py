#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

class IsaacService:
    def __init__(self,
        world,
        usd_visualizer):

        self._world = world
        self._usd_visualizer = usd_visualizer

    @property
    def world(self):
        return self._world
    
    @property
    def usd_visualizer(self):
        return self._usd_visualizer
