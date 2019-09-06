#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.registry import registry
from habitat.core.simulator import (
    Config,
    SensorSuite,
    Simulator,
    Space
)
import pyrobot
import numpy as np


# TODO(akadian): make MAX_DEPTH a parameter of the class
MAX_DEPTH = 5.0


@registry.register_simulator(name="PyRobot-v0")
class PyRobot(Simulator):
    def __init__(self, config: Config) -> None:
        config_pyrobot = {
            "base_controller": "proportional"
        }
        # self._forward_step_change = [0.25, 0, 0]

        self._angle_step = (10 / 180) * np.pi
        # self._left_step_change = [0, 0, self._angle_step]
        # self._right_step_change = [0, 0, -self._angle_step]

        self._robot = pyrobot.Robot("locobot", base_config=config_pyrobot)

    @property
    def sensor_suite(self) -> SensorSuite:
        return self._sensor_suite

    @property
    def action_space(self) -> Space:
        return self._action_space

    def reset(self):
        self._robot.camera.reset()
        rgb_observation = self.rgb_observation()
        depth_observation = self.depth_observation()

        return {
            "rgb": rgb_observation,
            "depth": depth_observation,
        }

    def _forward(self):
        target_position = [0.25, 0, 0]
        # TODO(akadian): see if there is a command to check if the step was successful
        self._robot.base.go_to_relative(
            target_position,
            smooth=False,
            close_loop=True
        )

    def _left(self):
        target_position = [0, 0, self._angle_step]
        self._robot.base.go_to_relative(
            target_position,
            smooth=False,
            close_loop=True
        )

    def _right(self):
        target_position = [0, 0, -self._angle_step]
        self._robot.base.go_to_relative(
            target_position,
            smooth=False,
            close_loop=True
        )

    def step(self, action):
        if action == 0:
            # forward
            self._forward()
        elif action == 1:
            # left
            self._left()
        elif action == 2:
            # right
            self._right()
        else:
            raise ValueError

        rgb_observation = self.rgb_observation()
        depth_observation = self.depth_observation()

        return {
            "rgb": rgb_observation,
            "depth": depth_observation,
        }

    # TODO(akadian): expose a method to just get the observations,
    #                this will be useful for the case if the state of robot
    #                changes due to external factors.

    def rgb_observation(self):
        rgb = self._robot.camera.get_rgb()
        return rgb

    def depth_observation(self):
        depth = self._robot.camera.get_depth()
        depth = depth / 1000  # convert from mm to m
        depth[depth > MAX_DEPTH] = MAX_DEPTH
        depth = depth / MAX_DEPTH  # scale to [0, 1]
        return depth

    def close(self):
        raise NotImplementedError
