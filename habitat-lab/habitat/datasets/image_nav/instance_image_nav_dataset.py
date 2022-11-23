#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from habitat.core.registry import registry
from habitat.core.simulator import AgentState
from habitat.core.utils import DatasetFloatJSONEncoder
from habitat.datasets.pointnav.pointnav_dataset import (
    DEFAULT_SCENE_PATH_PREFIX,
    PointNavDatasetV1,
)
from habitat.tasks.nav.instance_image_nav_task import (
    InstanceImageGoal,
    InstanceImageGoalNavEpisode,
    InstanceImageParameters,
)
from habitat.tasks.nav.object_nav_task import ObjectViewLocation

if TYPE_CHECKING:
    from omegaconf import DictConfig


@registry.register_dataset(name="InstanceImageNav-v1")
class InstanceImageNavDatasetV1(PointNavDatasetV1):
    """Class that loads an Instance Image Navigation dataset."""

    goals: Dict[str, InstanceImageGoal]
    episodes: List[InstanceImageGoalNavEpisode] = []  # type: ignore[assignment]

    def __init__(self, config: Optional["DictConfig"] = None) -> None:
        self.goals = {}
        super().__init__(config)

    def to_json(self) -> str:
        for i in range(len(self.episodes)):
            self.episodes[i].goals.clear()

        result = DatasetFloatJSONEncoder().encode(self)

        for i in range(len(self.episodes)):
            self.episodes[i].goals = [self.goals[self.episodes[i].goal_key]]

        return result

    @staticmethod
    def _deserialize_goal(
        serialized_goal: Dict[str, Any]
    ) -> InstanceImageGoal:
        g = InstanceImageGoal(**serialized_goal)

        for vidx, view in enumerate(g.view_points):
            view_location = ObjectViewLocation(**view)  # type: ignore[arg-type]
            view_location.agent_state = AgentState(**view_location.agent_state)  # type: ignore[arg-type]
            g.view_points[vidx] = view_location

        for iidx, params in enumerate(g.image_goals):
            g.image_goals[iidx] = InstanceImageParameters(**params)  # type: ignore[arg-type]

        return g

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)

        if len(deserialized["episodes"]) == 0:
            return

        assert "goals" in deserialized

        for k, g in deserialized["goals"].items():
            self.goals[k] = self._deserialize_goal(g)

        for episode in deserialized["episodes"]:
            episode = InstanceImageGoalNavEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.goals = [self.goals[episode.goal_key]]
            self.episodes.append(episode)  # type: ignore[attr-defined]
