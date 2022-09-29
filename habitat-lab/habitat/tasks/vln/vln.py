#!/usr/bin/env python3

# Owners/maintainers of the Vision and Language Navigation task:
#   @jacobkrantz: Jacob Krantz
#   @koshyanand: Anand Koshy

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional

import attr
from gym import spaces

from habitat.core.registry import registry
from habitat.core.simulator import Observations, Sensor
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import NavigationEpisode, NavigationTask


@attr.s(auto_attribs=True)
class InstructionData:
    instruction_text: str
    instruction_tokens: Optional[List[str]] = None


@attr.s(auto_attribs=True, kw_only=True)
class VLNEpisode(NavigationEpisode):
    r"""Specification of episode that includes initial position and rotation
    of agent, goal specifications, instruction specifications, reference path,
    and optional shortest paths.

    Args:
        episode_id: id of episode in the dataset
        scene_id: id of scene inside the simulator.
        start_position: numpy ndarray containing 3 entries for (x, y, z).
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation.
        goals: list of goals specifications
        reference_path: List of (x, y, z) positions which gives the reference
            path to the goal that aligns with the instruction.
        instruction: single natural language instruction guide to goal.
        trajectory_id: id of ground truth trajectory path.
    """
    reference_path: List[List[float]] = attr.ib(
        default=None, validator=not_none_validator
    )
    instruction: InstructionData = attr.ib(
        default=None, validator=not_none_validator
    )
    trajectory_id: int = attr.ib(default=None, validator=not_none_validator)


@registry.register_sensor(name="InstructionSensor")
class InstructionSensor(Sensor):
    def __init__(self, **kwargs):
        self.uuid = "instruction"
        self.observation_space = spaces.Discrete(0)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _get_observation(
        self,
        observations: Dict[str, Observations],
        episode: VLNEpisode,
        **kwargs
    ):
        return {
            "text": episode.instruction.instruction_text,
            "tokens": episode.instruction.instruction_tokens,
            "trajectory_id": episode.trajectory_id,
        }

    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)


@registry.register_task(name="VLN-v0")
class VLNTask(NavigationTask):
    r"""Vision and Language Navigation Task
    Goal: An agent must navigate to a goal location in a 3D environment
        specified by a natural language instruction.
    Metric: Success weighted by Path Length (SPL)
    Usage example:
        examples/vln_reference_path_follower_example.py
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
