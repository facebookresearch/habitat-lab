#!/usr/bin/env python3

import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import attr

from habitat.core.registry import registry
from habitat.core.simulator import AgentState
from habitat.core.utils import DatasetFloatJSONEncoder
from habitat.datasets.pointnav.pointnav_dataset import (
    CONTENT_SCENES_PATH_FIELD,
    DEFAULT_SCENE_PATH_PREFIX,
    PointNavDatasetV1,
)
from habitat.tasks.nav.goat_task import GoatEpisode
from habitat.tasks.nav.instance_image_nav_task import (  # InstanceImageGoalNavEpisode,
    InstanceImageGoal,
    InstanceImageParameters,
)
from habitat.tasks.nav.object_nav_task import (
    ObjectGoal,
    ObjectGoalNavEpisode,
    ObjectViewLocation,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


@attr.s(auto_attribs=True)
class OVONObjectViewLocation(ObjectViewLocation):
    r"""OVONObjectViewLocation

    Args:
        raidus: radius of the circle
    """
    radius: Optional[float] = None


@attr.s(auto_attribs=True, kw_only=True)
class LanguageNavEpisode(ObjectGoalNavEpisode):
    r"""OVON Episode

    :param children_object_categories: Category of the object
    """
    object_instance_id: Optional[int] = None
    instructions: Optional[List[str]] = []
    llm_response: Optional[Dict] = None

    @property
    def goals_key(self) -> str:
        r"""The key to retrieve the goals"""
        return f"{os.path.basename(self.scene_id)}_{self.object_instance_id}"


@registry.register_dataset(name="Goat-v1")
class GoatDatasetV1(PointNavDatasetV1):
    r"""
    Class inherited from PointNavDataset that loads GOAT dataset.
    """
    episodes: List[LanguageNavEpisode] = []  # type: ignore
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"
    goals: Dict[str, Sequence[ObjectGoal]]

    @staticmethod
    def dedup_goals(dataset: Dict[str, Any]) -> Dict[str, Any]:
        if len(dataset["episodes"]) == 0:
            return dataset

        goals = {}
        for i, ep in enumerate(dataset["episodes"]):
            ep = LanguageNavEpisode(**ep)

            goals_key = ep.goals_key
            if goals_key not in goals:
                goals[goals_key] = ep.goals

            dataset["episodes"][i]["goals"] = []

        dataset["goals"] = goals

        return dataset

    def to_json(self) -> str:
        for i in range(len(self.episodes)):
            self.episodes[i].goals = []

        result = DatasetFloatJSONEncoder().encode(self)

        for i in range(len(self.episodes)):
            goals = self.goals[self.episodes[i].goals_key]
            if not isinstance(goals, list):
                goals = list(goals)
            self.episodes[i].goals = goals

        return result

    def __init__(self, config: Optional["DictConfig"] = None) -> None:
        self.goals = {}
        super().__init__(config)
        self.episodes = list(self.episodes)

    @staticmethod
    def __deserialize_objectnav_goal(
        serialized_goal: Dict[str, Any]
    ) -> ObjectGoal:

        g = ObjectGoal(**serialized_goal)

        for vidx, view in enumerate(g.view_points):
            view_location = ObjectViewLocation(**view)  # type: ignore
            view_location.agent_state = AgentState(**view_location.agent_state)  # type: ignore
            g.view_points[vidx] = view_location

        return g

    @staticmethod
    def __deserialize_languagenav_goal(
        serialized_goal: Dict[str, Any]
    ) -> ObjectGoal:

        if serialized_goal.get("children_object_categories") is not None:
            del serialized_goal["children_object_categories"]

        g = ObjectGoal(**serialized_goal)

        for vidx, view in enumerate(g.view_points):
            view_location = OVONObjectViewLocation(**view)  # type: ignore
            view_location.agent_state = AgentState(**view_location.agent_state)  # type: ignore
            g.view_points[vidx] = view_location

        return g

    @staticmethod
    def __deserialize_imagenav_goal(
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
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        if len(deserialized["episodes"]) == 0:
            return

        if "goals" not in deserialized:
            deserialized = self.dedup_goals(deserialized)


        self.goals = deserialized["goals"]

        for i, composite_episode in enumerate(deserialized["episodes"]):
            composite_episode["goals"] = []
            composite_episode = GoatEpisode(**composite_episode)

            composite_episode.episode_id = str(i)

            if scenes_dir is not None:
                if composite_episode.scene_id.startswith(
                    DEFAULT_SCENE_PATH_PREFIX
                ):
                    composite_episode.scene_id = composite_episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                composite_episode.scene_id = os.path.join(
                    scenes_dir, "", composite_episode.scene_id
                )

            composite_episode.goals = []

            for goal in composite_episode.tasks:
                goal_type = goal[1]
                goal_category = goal[0]
                goal_inst_id = goal[2]

                dset_same_cat_goals = [x for x in self.goals.values() if x[0]['object_category'] == goal_category]

                assert len(dset_same_cat_goals) == 1, f"more than 1 goal categories for {goal_category}"

                if goal_type == "object":
                    composite_episode.goals.append(dset_same_cat_goals)
                else:
                    goal_inst = [x for x in dset_same_cat_goals[0] if x['object_id'] == goal_inst_id]
                    composite_episode.goals.append(goal_inst)

            self.episodes.append(composite_episode)  # type: ignore [attr-defined]
