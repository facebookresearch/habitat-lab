# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
from typing import TYPE_CHECKING, Any, List, Optional

import attr
import numpy as np
from gym import spaces

from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, Sensor, SensorTypes
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    NavigationTask,
)

from habitat.tasks.nav.object_nav_task import (
    ObjectGoal,
    ObjectGoalNavEpisode,
    ObjectViewLocation,
)

try:
    from habitat.datasets.object_nav.object_nav_dataset import (
        ObjectNavDatasetV1,
    )
except ImportError:
    pass

if TYPE_CHECKING:
    from omegaconf import DictConfig

@registry.register_sensor
class LanguageGoalSensor(Sensor):
    r"""A sensor for Language Goal specification as observations which is used in
    Language Navigation. The goal is expected to be specified by a caption.
    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalSensor sensor. Can contain field
            goal_spec that specifies which id use for goal specification,
            goal_spec_max_val the maximum object_id possible used for
            observation space definition.
        dataset: a Language navigation dataset that contains dictionaries
        of categories id to text mapping.
    """
    cls_uuid: str = "languagegoal"

    def __init__(
        self,
        sim,
        config: "DictConfig",
        dataset: "LanguageNavDatasetV1",
        *args: Any,
        **kwargs: Any,
    ):
        self._sim = sim
        self._dataset = dataset
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TEXT

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        # sensor_shape = (1,)
        # max_value = self.config.goal_spec_max_val - 1
        # if self.config.goal_spec == "TASK_CATEGORY_ID":
        #     max_value = max(
        #         self._dataset.category_to_task_category_id.values()
        #     )
        max_landmarks = 10
        max_rooms = 2
        return spaces.Dict({
            "category_name": spaces.Text(min_length=0, max_length=30),
            "caption": spaces.Text(min_length=0, max_length=200),
            "landmarks": spaces.Tuple([spaces.Text(min_length=0, max_length=30) for _ in range(max_landmarks)]),
            "rooms": spaces.Tuple([spaces.Text(min_length=0, max_length=30) for _ in range(max_rooms)]),
        })
    
    def extract_room_and_landmarks_from_response(self, response: str):
        landmarks = re.findall(r'LandmarkName\("([^"]+)"\)', response)
        rooms = re.findall(r'RoomName\("([^"]+)"\)', response)
        return rooms, landmarks

    def get_observation(
        self,
        observations,
        *args: Any,
        episode, # TODO
        **kwargs: Any,
    ) -> Optional[np.ndarray]:

        if len(episode.goals) == 0:
            logger.error(
                f"No goal specified for episode {episode.episode_id}."
            )
            return None
        if not isinstance(episode.goals[0], ObjectGoal):
            logger.error(
                f"First goal should be ObjectGoal, episode {episode.episode_id}."
            )
            return None

        category_name = episode.object_category
        caption = episode.instructions[0]
        rooms, landmarks = self.extract_room_and_landmarks_from_response(
            episode.llm_response["full_response"]
        )

        return {
            "category_name": category_name,
            "caption": caption,
            "room": rooms,
            "landmarks": landmarks,
        }
        # if self.config.goal_spec == "TASK_CATEGORY_ID":
        #     return np.array(
        #         [self._dataset.category_to_task_category_id[category_name]],
        #         dtype=np.int64,
        #     )
        # elif self.config.goal_spec == "OBJECT_ID":
        #     obj_goal = episode.goals[0]
        #     assert isinstance(obj_goal, ObjectGoal)  # for type checking
        #     return np.array([obj_goal.object_name_id], dtype=np.int64)
        # else:
        #     raise RuntimeError(
        #         "Wrong goal_spec specified for ObjectGoalSensor."
        #     )


@registry.register_task(name="LanguageNav-v1")
class LanguageNavigationTask(NavigationTask): # TODO
    r"""A Language Navigation Task class for a task specific methods.
    Used to explicitly state a type of the task in config.
    """
