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
from habitat.datasets.utils import check_and_gen_physics_config
from habitat.tasks.nav.object_nav_task import ObjectGoal, ObjectViewLocation

if TYPE_CHECKING:
    from omegaconf import DictConfig


@attr.s(auto_attribs=True, kw_only=True)
class RearrangeEpisode(Episode):
    r"""Specifies additional objects, targets, markers, and ArticulatedObject states for a particular instance of an object rearrangement task.

    :property ao_states: Lists modified ArticulatedObject states for the scene: {instance_handle -> {link, state}}
    :property rigid_objs: A list of objects to add to the scene, each with: (handle, transform)
    :property targets: Maps an object instance to a new target location for placement in the task. {instance_name -> target_transform}
    :property markers: Indicate points of interest in the scene such as grasp points like handles. {marker name -> (type, (params))}
    :property target_receptacles: The names and link indices of the receptacles containing the target objects.
    :property goal_receptacles: The names and link indices of the receptacles containing the goals.
    """
    ao_states: Dict[str, Dict[int, float]]
    rigid_objs: List[Tuple[str, np.ndarray]]
    targets: Dict[str, np.ndarray]
    markers: List[Dict[str, Any]] = []
    target_receptacles: List[Tuple[str, int]] = []
    goal_receptacles: List[Tuple[str, int]] = []
    name_to_receptacle: Dict[str, str] = {}


@attr.s(auto_attribs=True, kw_only=True)
class ObjectRearrangeEpisode(RearrangeEpisode):
    r"""Specifies categories of the object, start and goal receptacles

    :property object_category: Category of the object to be rearranged
    :property start_recep_category: Category of the start receptacle
    :property goal_recep_category: Category of the goal receptacle
    """
    object_category: Optional[str] = None
    start_recep_category: Optional[str] = None
    goal_recep_category: Optional[str] = None
    candidate_objects: Optional[List[ObjectGoal]] = None
    candidate_start_receps: Optional[List[ObjectGoal]] = None
    candidate_goal_receps: Optional[List[ObjectGoal]] = None


@registry.register_dataset(name="RearrangeDataset-v0")
class RearrangeDatasetV0(PointNavDatasetV1):
    r"""Class inherited from PointNavDataset that loads Rearrangement dataset."""
    episodes: List[RearrangeEpisode] = []  # type: ignore
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

    def to_json(self) -> str:
        result = DatasetFloatJSONEncoder().encode(self)
        return result

    def __init__(self, config: Optional["DictConfig"] = None) -> None:
        self.config = config

        if config and not self.check_config_paths_exist(config):
            logger.info(
                "Rearrange task assets are not downloaded locally, downloading and extracting now..."
            )
            data_downloader.main(
                ["--uids", "rearrange_task_assets", "--no-replace"]
            )
            logger.info("Downloaded and extracted the data.")

        check_and_gen_physics_config()

        super().__init__(config)

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)

        for i, episode in enumerate(deserialized["episodes"]):
            rearrangement_episode = RearrangeEpisode(**episode)
            rearrangement_episode.episode_id = str(i)

            self.episodes.append(rearrangement_episode)


@registry.register_dataset(name="ObjectRearrangeDataset-v0")
class ObjectRearrangeDatasetV0(PointNavDatasetV1):
    r"""Class inherited from PointNavDataset that loads Object Rearrangement dataset."""
    obj_category_to_obj_category_id: Dict[str, int]
    recep_category_to_recep_category_id: Dict[str, int]
    episodes: List[ObjectRearrangeEpisode] = []  # type: ignore
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

    def to_json(self) -> str:
        result = DatasetFloatJSONEncoder().encode(self)
        return result

    def __init__(self, config: Optional["DictConfig"] = None) -> None:
        self.config = config
        check_and_gen_physics_config()

        super().__init__(config)

    @staticmethod
    def __deserialize_goal(serialized_goal: Dict[str, Any]) -> ObjectGoal:
        g = ObjectGoal(**serialized_goal)

        for vidx, view in enumerate(g.view_points):
            view_location = ObjectViewLocation(**view)  # type: ignore
            view_location.agent_state = AgentState(**view_location.agent_state)  # type: ignore
            g.view_points[vidx] = view_location
        return g

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)

        if "obj_category_to_obj_category_id" in deserialized:
            self.obj_category_to_obj_category_id = deserialized[
                "obj_category_to_obj_category_id"
            ]
        if "recep_category_to_recep_category_id" in deserialized:
            self.recep_category_to_recep_category_id = deserialized[
                "recep_category_to_recep_category_id"
            ]

        for i, episode in enumerate(deserialized["episodes"]):
            rearrangement_episode = ObjectRearrangeEpisode(**episode)
            rearrangement_episode.episode_id = str(i)
            for goal_type in [
                "candidate_objects",
                "candidate_start_receps",
                "candidate_goal_receps",
            ]:
                if goal_type in episode:
                    setattr(
                        rearrangement_episode,
                        goal_type,
                        [
                            self.__deserialize_goal(g)
                            for g in episode[goal_type]
                        ],
                    )
            self.episodes.append(rearrangement_episode)
