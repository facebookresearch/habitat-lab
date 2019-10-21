#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import attr
import numpy as np
from gym import spaces

from habitat.core.registry import registry
from habitat.core.simulator import (
    Observations,
    Sensor,
    SensorSuite,
    SensorTypes,
)
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import NavigationEpisode, NavigationTask, NavigationGoal
from typing import Any, List, Optional, Type
import habitat
from habitat.core.dataset import Episode

@attr.s(auto_attribs=True, kw_only=True)
class VLNGoal(NavigationGoal):
    r"""VLNGoal that can be specified by instructions, goal position and radius
    """
    instruction: str = attr.ib(
        default=None, validator=not_none_validator
    )

@attr.s(auto_attribs=True, kw_only=True)
class VLNEpisode(Episode):
    r"""Specification of episode that includes initial position and rotation of
    agent, goal specifications and optional shortest paths.

    Args:
        scene_id: id of scene inside the simulator.
        start_position: numpy ndarray containing 3 entries for (x, y, z).
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation.
        goals: relevant goal object/room.
        Instructions: question related to goal object.
    """
    path: List[List[float]] = attr.ib(
        default=None, validator=not_none_validator
    )
    goals: List[Optional[VLNGoal]] = None


@registry.register_sensor(name="InstructionSensor")
class InstructionSensor(Sensor):
    def __init__(self, **kwargs):
        self.uuid = "instruction"
        self.sensor_type = SensorTypes.TEXT
        # TODO (maksymets) extend gym observation space for text and metadata
        self.observation_space = spaces.Discrete(0)
    
    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return self.sensor_type

    def _get_observation(
        self,
        observations: Dict[str, Observations],
        episode: NavigationEpisode,
        **kwargs
    ):
        return episode.goals[0].instruction

    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)

# # TODO (maksymets) Move reward to measurement class
# class RewardSensor(Sensor):
#     REWARD_MIN = -100
#     REWARD_MAX = -100

#     def __init__(self, **kwargs):
#         self.uuid = "reward"
#         self.sensor_type = SensorTypes.TENSOR
#         self.observation_space = spaces.Box(
#             low=RewardSensor.REWARD_MIN,
#             high=RewardSensor.REWARD_MAX,
#             shape=(1,),
#             dtype=np.float,
#         )

#     def _get_observation(
#         self,
#         observations: Dict[str, Observations],
#         episode: NavigationEpisode,
#         **kwargs
#     ):
#         return [0]

#     def get_observation(self, **kwargs):
#         return self._get_observation(**kwargs)

@registry.register_task(name="VLN-v0")
class VLNTask(NavigationTask):
    _sensor_suite: SensorSuite

    def __init__(self, **kwargs) -> None:
        self._sensor_suite = SensorSuite(
            [InstructionSensor()]
        )
        super().__init__(**kwargs)

