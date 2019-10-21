#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import attr
from gym import Space, spaces

from habitat.core.registry import registry
from habitat.core.simulator import (
    Observations,
    Sensor,
    SensorSuite,
    SensorTypes,
)
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import (
	NavigationEpisode,
	NavigationTask,
	NavigationGoal
)
from typing import Dict, Optional, Any, List, Optional, Type


@attr.s(auto_attribs=True, kw_only=True)
class VLNEpisode(NavigationEpisode):
    r"""Specification of episode that includes initial position and rotation of
    agent, goal specifications, instruction specifications, and optional shortest paths.

    Args:
        scene_id: id of scene inside the simulator.
        start_position: numpy ndarray containing 3 entries for (x, y, z).
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation.
        instruction: single instruction guide to goal.
        trajectory_id: id of ground truth trajectory path.
        goals: relevant goal object/room.
    """
    path: List[List[float]] = attr.ib(
        default=None, validator=not_none_validator
    )
    instruction: str = attr.ib(
        default=None, validator=not_none_validator
    )
    trajectory_id: int = attr.ib(
        default=None, validator=not_none_validator
    )
    goals: List[NavigationGoal] = None


@registry.register_sensor(name="InstructionSensor")
class InstructionSensor(Sensor):
    def __init__(self, **kwargs):
        self.uuid = "instruction"
        # when InstructionData exists:
        #   SensorTypes.TOKEN_IDS
        self.sensor_type = SensorTypes.TEXT
        self.observation_space = spaces.Discrete(0)
    
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return self.sensor_type

    def _get_observation(
        self,
        observations: Dict[str, Observations],
        episode: VLNEpisode,
        **kwargs
    ):
        return {
            "text": episode.instruction,
            "trajectory_id": episode.trajectory_id
        }

    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)
    
    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        pass
        # when InstructionData exists:
        # return ListSpace(
        #     spaces.Discrete(self._dataset.instruction_vocab.get_size())
        # )


@registry.register_task(name="VLN-v0")
class VLNTask(NavigationTask):
    _sensor_suite: SensorSuite

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._sensor_suite = SensorSuite(
            [InstructionSensor()]
        )
