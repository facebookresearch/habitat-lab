
import attr
import gzip
import json
import os
from typing import List, Optional
import numpy as np

from habitat.config import read_write
from habitat.core.utils import not_none_validator
from habitat.core.dataset import ALL_SCENES_MASK, Dataset
from habitat.core.registry import registry
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    ShortestPathPoint,
)
from habitat.tasks.nav.instance_image_nav_task import (
    InstanceImageGoal,
    InstanceImageGoalNavEpisode,
    InstanceImageParameters,
)
from habitat.datasets.pointnav.pointnav_dataset import (
    PointNavDatasetV1, 
    CONTENT_SCENES_PATH_FIELD,
    DEFAULT_SCENE_PATH_PREFIX,
) 
from habitat.datasets.image_nav.instance_image_nav_dataset import InstanceImageNavDatasetV1


@attr.s(auto_attribs=True, kw_only=True)
class NavigationGoalV2(NavigationGoal):
    r"""Base class for a goal specification hierarchy."""

    position: List[float] = attr.ib(default=None, validator=not_none_validator)
    rotation: Optional[List[float]] = None
    radius: Optional[float] = None


@registry.register_dataset(name="ImageNav-v1")
class ImageNavDatasetV1(InstanceImageNavDatasetV1):
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
            goal_key = InstanceImageGoalNavEpisode(**episode).goal_key

            episode.pop('object_category')
            episode.pop('goal_object_id')
            episode.pop('goal_image_id')
            episode = NavigationEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)
            
            episode.goals = [self._select_view(self.goals[goal_key], episode.episode_id)]
            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        path[p_index] = ShortestPathPoint(**point)

            self.episodes.append(episode)  # typ

    def _select_view(self, goal: InstanceImageGoal, episode_id: str):
        seed = abs(hash(episode_id)) % (2**32)
        rng = np.random.RandomState(seed)
        selected_view = rng.choice(goal.view_points)

        goal = selected_view.agent_state
        return NavigationGoalV2(position=goal.position, rotation=goal.rotation)
        