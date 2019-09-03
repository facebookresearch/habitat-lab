#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from habitat.core.simulator import (
    Config,
    SensorSuite,
    Simulator,
    Space
)


class PyRobot(Simulator):
    def __init__(self, config: Config) -> None:
        raise NotImplementedError

    @property
    def sensor_suite(self) -> SensorSuite:
        return self._sensor_suite

    @property
    def action_space(self) -> Space:
        return self._action_space

    def step(self, action):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
