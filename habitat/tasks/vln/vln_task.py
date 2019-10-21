#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import attr
import numpy as np
from gym import Space, spaces

from habitat.core.registry import registry
from habitat.core.simulator import (
    Observations,
    Sensor,
    SensorSuite,
    SensorTypes,
)
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import NavigationGoal, NavigationEpisode, NavigationTask
from typing import Any, List, Optional, Type



@attr.s(auto_attribs=True, kw_only=True)
class VLNGoal(NavigationGoal):
    r"""VLNGoal that can be specified by instructions, goal position and radius
    """
    instructions: List[str] = attr.ib(default=None, validator=not_none_validator)

@attr.s(auto_attribs=True, kw_only=True)
class VLNEpisode(NavigationEpisode):
    r"""Specification of episode that includes initial position and rotation of
    agent, goal, question specifications and optional shortest paths.

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
    instructions: List[str] = attr.ib(
        default=None, validator=not_none_validator
    )

@registry.register_sensor(name="InstructionSensor")
class InstructionSensor(Sensor):
    def __init__(self, **kwargs):
        self.uuid = "instruction"
        # when InstructionData exists:
        #   SensorTypes.TOKEN_IDS
        self.sensor_type = SensorTypes.TEXT
        self.observation_space = spaces.Discrete(0)
        self.previous_instruction = {
            "episode_id": -1,
            "instruction": ""
        }

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "instruction"

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return self.sensor_type

    def _get_observation(
        self,
        observations: Dict[str, Observations],
        episode: VLNEpisode,
        **kwargs
    ):
        # when InstructionData exists: intruction.instruction_tokens
        if self.previous_instruction["episode_id"] == episode.episode_id:
            return self.previous_instruction["instruction"]

        self.previous_instruction = {
            "episode_id": episode.episode_id,
            "instruction": np.random.choice(episode.instructions)
        }
        return self.previous_instruction["instruction"]

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
