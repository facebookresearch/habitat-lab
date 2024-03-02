#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import attr
import numpy as np

import habitat_sim.utils.datasets_download as data_downloader
from habitat.core.dataset import Episode
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.utils import DatasetFloatJSONEncoder
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.datasets.utils import check_and_gen_physics_config

if TYPE_CHECKING:
    from omegaconf import DictConfig


class AgentEpisode():
    start_position: List[float]
    goal_position: List[float]
    
    def get_door_start_goal(self, ):
        sem_scene = self.sim.semantic_annotations() #TODO: check inside semantic
        #TODO
        return (start_position, goal_position)


@attr.s(auto_attribs=True, kw_only=True)
class SocialNavigationEpisode(Episode):
    r"""Class for episode specification that includes initial position and
    rotation of agent, scene name, goal and optional shortest paths. An
    episode is a description of one task instance for the agent.

    Args:
        episode_id: id of episode in the dataset, usually episode number
        scene_id: id of scene in scene dataset
        start_position: numpy ndarray containing 3 entries for (x, y, z)
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation. ref: https://en.wikipedia.org/wiki/Versor
        goals: list of goals specifications
        start_room: room id
        shortest_paths: list containing shortest paths to goals
        ao_states: lists modified ArticulatedObject states for the scene: {instance_handle -> {link, state}}
    """

    goals: List[NavigationGoal] = attr.ib(
        default=None,
        validator=not_none_validator,
        on_setattr=Episode._reset_shortest_path_cache_hook,
    )
    start_room: Optional[str] = None
    shortest_paths: Optional[List[List[ShortestPathPoint]]] = None
    agents: List[str, AgentEpisode]


# Use PointNavDatasetV1 from pointnav_dataset

