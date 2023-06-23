#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

import attr
import numpy as np

import habitat_sim.utils.datasets_download as data_downloader
from habitat.core.dataset import Episode, EpisodeIterator
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import AgentState
from habitat.core.utils import DatasetFloatJSONEncoder
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat.datasets.utils import check_and_gen_physics_config
from habitat.tasks.nav.object_nav_task import ObjectGoal, ObjectViewLocation
from habitat.datasets.rearrange.rearrange_dataset import RearrangeEpisode

if TYPE_CHECKING:
    from omegaconf import DictConfig



@attr.s(auto_attribs=True, kw_only=True)
class OVMMEpisode(RearrangeEpisode):
    r"""Specifies categories of the object, start and goal receptacles

    :property object_category: Category of the object to be rearranged
    :property start_recep_category: Category of the start receptacle
    :property goal_recep_category: Category of the goal receptacle
    """
    object_category: Optional[str] = None
    start_recep_category: Optional[str] = None
    goal_recep_category: Optional[str] = None
    candidate_objects: Optional[List[ObjectGoal]] = None
    candidate_objects_hard: Optional[List[ObjectGoal]] = None
    candidate_start_receps: Optional[List[ObjectGoal]] = None
    candidate_goal_receps: Optional[List[ObjectGoal]] = None



class OVMMEpisodeIterator(EpisodeIterator[OVMMEpisode]):
    def __init__(
        self,
        viewpoints_matrix,
        transformations_matrix,
        episodes,
        *args,
        **kwargs
    ):
        self.viewpoints = viewpoints_matrix
        self.transformations = transformations_matrix
        self._vp_keys = [
            "candidate_objects",
            "candidate_objects_hard",
            "candidate_goal_receps",
        ]
        super().__init__(episodes, *args, **kwargs)

    def __next__(self) -> OVMMEpisode:
        # deepcopy is to avoid increasing memory as we iterate through the episodes
        episode = cast(
            OVMMEpisode, copy.deepcopy(super().__next__())
        )

        deserialized_objs = []
        if self.transformations is not None:
            for rigid_obj in episode.rigid_objs:
                transform = np.vstack(
                    (self.transformations[rigid_obj[1]], [0, 0, 0, 1])
                )
                deserialized_objs.append((rigid_obj[0], transform))
            episode.rigid_objs = deserialized_objs

        if self.viewpoints is None:
            return episode

        for vp_key in self._vp_keys:
            obj_goal: ObjectGoal
            for obj_goal in getattr(episode, vp_key):
                for vidx, view_idx in enumerate(obj_goal.view_points):
                    view = self.viewpoints[view_idx]
                    position, rotation, iou = (
                        view[:3],
                        view[3:7],
                        view[7].item(),
                    )
                    agent_state = AgentState(position, rotation)
                    obj_goal.view_points[vidx] = ObjectViewLocation(
                        agent_state, iou
                    )

        return episode


@registry.register_dataset(name="OVMMDataset-v0")
class OVMMDatasetV0(PointNavDatasetV1):
    r"""Class inherited from PointNavDataset that loads Object Rearrangement dataset."""
    obj_category_to_obj_category_id: Dict[str, int]
    recep_category_to_recep_category_id: Dict[str, int]
    episodes: List[OVMMEpisode] = []  # type: ignore
    viewpoints_matrix: np.ndarray = None
    transformations_matrix: np.ndarray = None
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"


    def __init__(self, config: Optional["DictConfig"] = None) -> None:
        self.config = config
        check_and_gen_physics_config()
        self.episode_indices_range = None
        if config is not None:
            if self.config.viewpoints_matrix_path is not None:
                self.viewpoints_matrix = np.load(
                    self.config.viewpoints_matrix_path.format(
                        split=self.config.split
                    )
                )
            if self.config.transformations_matrix_path is not None:
                self.transformations_matrix = np.load(
                    self.config.transformations_matrix_path.format(
                        split=self.config.split
                    )
                )
            self.episode_indices_range = self.config.episode_indices_range
        super().__init__(config)

    def get_episode_iterator(
        self, *args: Any, **kwargs: Any
    ) -> OVMMEpisodeIterator:
        return OVMMEpisodeIterator(
            self.viewpoints_matrix,
            self.transformations_matrix,
            self.episodes,
            *args,
            **kwargs
        )

    def to_json(self) -> str:
        result = DatasetFloatJSONEncoder().encode(self)
        return result

    def __deserialize_goal(
        self, serialized_goal: Dict[str, Any]
    ) -> ObjectGoal:
        g = ObjectGoal(**serialized_goal)
        if self.viewpoints_matrix is None:
            # if the view points are not cached separately, read from original episodes
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


        all_episodes = deserialized["episodes"]
        if self.episode_indices_range is None:
            episodes_index_low, episodes_index_high = 0, len(all_episodes)
        else:
            episodes_index_low, episodes_index_high = self.episode_indices_range
        episode_ids_subset = None
        if len(self.config.episode_ids) > 0:
            episode_ids_subset = self.config.episode_ids[episodes_index_low: episodes_index_high]
        else:
            all_episodes = all_episodes[episodes_index_low: episodes_index_high]
        for i, episode in enumerate(all_episodes):
            rearrangement_episode = OVMMEpisode(**episode)
            rearrangement_episode.episode_id = str(i)
            for goal_type in [
                "candidate_objects",
                "candidate_objects_hard",
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

            if (
                episode_ids_subset is None
                or int(episode['episode_id']) in episode_ids_subset
            ):
                self.episodes.append(rearrangement_episode)
