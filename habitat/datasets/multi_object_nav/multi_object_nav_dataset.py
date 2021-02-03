#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from typing import Any, Dict, List, Optional

from habitat.config import Config
from habitat.core.registry import registry
from habitat.datasets.pointnav.pointnav_dataset import (
    CONTENT_SCENES_PATH_FIELD,
    DEFAULT_SCENE_PATH_PREFIX,
    PointNavDatasetV1,
)
from habitat.tasks.nav.multi_object_nav_task import (
    MultiObjectGoal,
    MultiObjectGoalNavEpisode,
)


@registry.register_dataset(name="MultiObjectNav-v1")
class MultiObjectNavDatasetV1(PointNavDatasetV1):
    r"""Class inherited from PointNavDataset that loads MultiON dataset."""
    category_to_task_category_id: Dict[str, int]
    episodes: List[Any] = []
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

    def __init__(self, config: Optional[Config] = None) -> None:
        super().__init__(config)

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

        if len(deserialized["episodes"]) == 0:
            return

        for i, episode in enumerate(deserialized["episodes"]):
            episode = MultiObjectGoalNavEpisode(**episode)
            episode.episode_id = str(i)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.goals = [MultiObjectGoal(**i) for i in episode.goals]

            self.episodes.append(episode)
