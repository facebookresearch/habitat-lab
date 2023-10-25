#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from gym import spaces

from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.rearrange.utils import UsesArticulatedAgentInterface


@registry.register_sensor
class SpotHeadStereoDepthSensor(UsesArticulatedAgentInterface, Sensor):
    """For Spot only. Sensor fusion for inputs of Spot stereo pair depth sensor.
    We want to combine head stereo depth images along with resizing it so that its size is the same as the size of the arm depth image.
    Spot's arm depth size: (240, 228, 1)
    Spot's head stereo depth size: (212, 120, 1)
    """

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim
        self._height = config.height
        self._width = config.width

    def _get_uuid(self, *args, **kwargs):
        return "spot_head_stereo_depth_sensor"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(
                config.height,
                config.width,
                1,
            ),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, task, *args, **kwargs):
        # Assert Spot's mobile gaze policy input
        require_sensors = ["head_stereo_right_depth", "head_stereo_left_depth"]
        if self.agent_id is None:
            target_key = require_sensors
        else:
            target_key = [
                f"agent_{self.agent_id}_{key}" for key in require_sensors
            ]
        for key in target_key:
            assert key in observations

        # Combine two images
        stereo_img = [observations[key] for key in target_key]
        stereo_img = np.concatenate(stereo_img, axis=1)

        # Resize the image from (212, 240, 1) to (240, 228, 1)
        # Zero padding in the first dim (height). 14 is from (240-212)/2
        stereo_img = np.pad(
            stereo_img,
            ((14, 14), (0, 0), (0, 0)),
            "constant",
            constant_values=0,
        )
        # Cut the second dim (width). 6 is from (240-228)/2
        stereo_img = stereo_img[:, 6:-6, :]

        return stereo_img
