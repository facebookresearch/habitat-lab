# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class IsaacService:
    def __init__(self, simulation_app, world, usd_visualizer):
        self._simulation_app = simulation_app
        self._world = world
        self._usd_visualizer = usd_visualizer

    @property
    def simulation_app(self):
        return self._simulation_app

    @property
    def world(self):
        return self._world

    @property
    def usd_visualizer(self):
        return self._usd_visualizer
