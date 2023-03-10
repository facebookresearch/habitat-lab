#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, Any

import numpy as np
import pyrobot
from gym import Space, spaces

from habitat.core.registry import registry
from habitat.core.simulator import (
    BumpSensor,
    DepthSensor,
    RGBSensor,
    SensorSuite,
    Simulator,
)
from habitat.core.utils import center_crop, try_cv2_import

if TYPE_CHECKING:
    from omegaconf import DictConfig


cv2 = try_cv2_import()


def _locobot_base_action_space():
    return spaces.Dict(
        {
            "go_to_relative": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
            "go_to_absolute": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
        }
    )


def _locobot_camera_action_space():
    return spaces.Dict(
        {
            "set_pan": spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
            "set_tilt": spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
            "set_pan_tilt": spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
        }
    )


def _resize_observation(obs, observation_space, config):
    if obs.shape != observation_space.shape:
        if (
            config.center_crop is True
            and obs.shape[0] > observation_space.shape[0]
            and obs.shape[1] > observation_space.shape[1]
        ):
            obs = center_crop(obs, observation_space)

        else:
            obs = cv2.resize(
                obs, (observation_space.shape[1], observation_space.shape[0])
            )
    return obs


MM_IN_METER = 1000  # millimeters in a meter
ACTION_SPACES = {
    "locobot": {
        "base_actions": _locobot_base_action_space(),
        "camera_actions": _locobot_camera_action_space(),
    }
}


@registry.register_sensor
class PyRobotRGBSensor(RGBSensor):
    def __init__(self, config):
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.config.height, self.config.width, 3),
            dtype=np.uint8,
        )

    def get_observation(self, robot_obs, *args: Any, **kwargs: Any):
        obs = robot_obs.get(self.uuid, None)

        assert obs is not None, "Invalid observation for {} sensor".format(
            self.uuid
        )

        obs = _resize_observation(obs, self.observation_space, self.config)

        return obs


@registry.register_sensor
class PyRobotDepthSensor(DepthSensor):
    min_depth_value: float
    max_depth_value: float

    def __init__(self, config):
        if config.normalize_depth:
            self.min_depth_value = 0
            self.max_depth_value = 1
        else:
            self.min_depth_value = config.min_depth
            self.max_depth_value = config.max_depth

        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=self.min_depth_value,
            high=self.max_depth_value,
            shape=(self.config.height, self.config.width, 1),
            dtype=np.float32,
        )

    def get_observation(self, robot_obs, *args: Any, **kwargs: Any):
        obs = robot_obs.get(self.uuid, None)

        assert obs is not None, "Invalid observation for {} sensor".format(
            self.uuid
        )

        obs = _resize_observation(obs, self.observation_space, self.config)

        obs = obs / MM_IN_METER  # convert from mm to m

        obs = np.clip(obs, self.config.min_depth, self.config.max_depth)
        if self.config.normalize_depth:
            # normalize depth observations to [0, 1]
            obs = (obs - self.config.min_depth) / (
                self.config.max_depth - self.config.min_depth
            )

        obs = np.expand_dims(obs, axis=2)  # make depth observations a 3D array

        return obs


@registry.register_sensor
class PyRobotBumpSensor(BumpSensor):
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=False, high=True, shape=(1,), dtype=bool)

    def get_observation(self, robot_obs, *args: Any, **kwargs: Any):
        return np.array(robot_obs["bump"])


@registry.register_simulator(name="PyRobot-v0")
class PyRobot(Simulator):
    r"""Simulator wrapper over PyRobot.

    PyRobot repo: https://github.com/facebookresearch/pyrobot
    To use this abstraction the user will have to setup PyRobot
    python3 version. Please refer to the PyRobot repository
    for setting it up. The user will also have to export a
    ROS_PATH environment variable to use this integration,
    please refer to :ref:`habitat.core.utils.try_cv2_import` for
    more details on this.

    This abstraction assumes that reality is a simulation
    (https://www.youtube.com/watch?v=tlTKTTt47WE).

    Args:
        config: configuration for initializing the PyRobot object.
    """

    def __init__(self, config: "DictConfig") -> None:
        self._config = config

        robot_sensors = []
        for sensor_cfg in self._config.sensors.values():
            sensor_type = registry.get_sensor(sensor_cfg.type)

            assert sensor_type is not None, "invalid sensor type {}".format(
                sensor_cfg.type
            )
            robot_sensors.append(sensor_type(sensor_cfg))
        self._sensor_suite = SensorSuite(robot_sensors)

        config_pyrobot = {
            "base_controller": self._config.base_controller,
            "base_planner": self._config.base_planner,
        }

        assert (
            self._config.robot in self._config.robots
        ), "Invalid robot type {}".format(self._config.robot)
        self._robot_config = getattr(self._config, self._config.robot)

        self._action_space = self._robot_action_space(
            self._config.robot, self._robot_config
        )

        self._robot = pyrobot.Robot(
            self._config.robot, base_config=config_pyrobot
        )

    def get_robot_observations(self):
        return {
            "rgb": self._robot.camera.get_rgb(),
            "depth": self._robot.camera.get_depth(),
            "bump": self._robot.base.base_state.bumper,
        }

    @property
    def sensor_suite(self) -> SensorSuite:
        return self._sensor_suite

    @property
    def base(self):
        return self._robot.base

    @property
    def camera(self):
        return self._robot.camera

    def _robot_action_space(self, articulated_agent_type, robot_config):
        action_spaces_dict = {}
        for action in robot_config.actions:
            action_spaces_dict[action] = ACTION_SPACES[articulated_agent_type][
                action
            ]
        return spaces.Dict(action_spaces_dict)

    @property
    def action_space(self) -> Space:
        return self._action_space

    def reset(self):
        self._robot.camera.reset()

        observations = self._sensor_suite.get_observations(
            robot_obs=self.get_robot_observations()
        )
        return observations

    def step(self, action, action_params):
        r"""Step in reality. Currently the supported
        actions are the ones defined in :ref:`_locobot_base_action_space`
        and :ref:`_locobot_camera_action_space`. For details on how
        to use these actions please refer to the documentation
        of namesake methods in PyRobot
        (https://github.com/facebookresearch/pyrobot).
        """
        if action in self._robot_config.base_actions:
            getattr(self._robot.base, action)(**action_params)
        elif action in self._robot_config.camera_actions:
            getattr(self._robot.camera, action)(**action_params)
        else:
            raise ValueError("Invalid action {}".format(action))

        observations = self._sensor_suite.get_observations(
            robot_obs=self.get_robot_observations()
        )

        return observations

    def render(self, mode: str = "rgb") -> Any:
        observations = self._sensor_suite.get_observations(
            robot_obs=self.get_robot_observations()
        )

        output = observations.get(mode)
        assert output is not None, "mode {} sensor is not active".format(mode)

        return output

    def get_agent_state(
        self, agent_id: int = 0, base_state_type: str = "odom"
    ):
        assert agent_id == 0, "No support of multi agent in {} yet.".format(
            self.__class__.__name__
        )
        state = {
            "base": self._robot.base.get_state(base_state_type),
            "camera": self._robot.camera.get_state(),
        }
        # TODO(akadian): add arm state when supported
        return state

    def seed(self, seed: int) -> None:
        raise NotImplementedError("No support for seeding in reality")
