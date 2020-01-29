#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from typing import Dict, List, Optional

from habitat.core.registry import registry
from habitat.core.simulator import AgentState, ShortestPathPoint
from habitat.core.utils import DatasetFloatJSONEncoder
from habitat.datasets.pointnav.pointnav_dataset import (
    CONTENT_SCENES_PATH_FIELD,
    DEFAULT_SCENE_PATH_PREFIX,
    PointNavDatasetV1,
)
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.tasks.nav.object_nav_task import ObjectGoal, ObjectViewLocation


@registry.register_dataset(name="ObjectNav-v1")
class ObjectNavDatasetV1(PointNavDatasetV1):
    r"""Class inherited from PointNavDataset that loads Object Navigation dataset.
    """
    category_to_task_category_id: Dict[str, int]
    category_to_mp3d_category_id: Dict[str, int]
    episodes: List[NavigationEpisode]
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

    def to_json(self) -> str:
        result = DatasetFloatJSONEncoder().encode(self)
        return result

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        if "category_to_task_category_id" in deserialized:
            self.category_to_task_category_id = deserialized[
                "category_to_task_category_id"
            ]

        if "category_to_mp3d_category_id" in deserialized:
            self.category_to_mp3d_category_id = deserialized[
                "category_to_mp3d_category_id"
            ]

        assert len(self.category_to_task_category_id) == len(
            self.category_to_mp3d_category_id
        )

        assert set(self.category_to_task_category_id.keys()) == set(
            self.category_to_mp3d_category_id.keys()
        ), "category_to_task and category_to_mp3d must have the same keys"

        for episode in deserialized["episodes"]:
            episode = NavigationEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            for i in range(len(episode.goals)):
                episode.goals[i] = ObjectGoal(**episode.goals[i])

                for vidx, view in enumerate(episode.goals[i].view_points):
                    view_location = ObjectViewLocation(**view)
                    view_location.agent_state = AgentState(
                        **view_location.agent_state
                    )
                    episode.goals[i].view_points[vidx] = view_location

            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        point = {
                            "action": point,
                            "rotation": None,
                            "position": None,
                        }
                        path[p_index] = ShortestPathPoint(**point)

            self.episodes.append(episode)

        for i, ep in enumerate(self.episodes):
            ep.episode_id = str(i)
