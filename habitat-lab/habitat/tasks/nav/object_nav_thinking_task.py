# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import cv2
import attr
import numpy as np
from gym import spaces

from habitat.core.dataset import Dataset, Episode
from habitat.core.simulator import Observations
from habitat.core.embodied_task import EmbodiedTask, SimulatorTaskAction
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    Sensor,
    SensorTypes,
    Simulator,
)
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    NavigationTask,
)
from habitat.tasks.nav.object_nav_task import (
    ObjectGoal,
    ObjectGoalNavEpisode,
    ObjectNavigationTask,
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


@attr.s(auto_attribs=True, kw_only=True)
class ObjectGoalNavThinkingEpisode(NavigationEpisode):
    r"""ObjectGoalThinking Navigation Episode

    :param object_category: Category of the obect
    """
    object_category: Optional[str] = None

    @property
    def goals_key(self) -> str:
        r"""The key to retrieve the goals"""
        return f"{os.path.basename(self.scene_id)}_{self.object_category}"


@registry.register_sensor
class ObjectGoalSensor(Sensor):
    r"""A sensor for Object Goal specification as observations which is used in
    ObjectGoal Navigation. The goal is expected to be specified by object_id or
    semantic category id.
    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalSensor sensor. Can contain field
            goal_spec that specifies which id use for goal specification,
            goal_spec_max_val the maximum object_id possible used for
            observation space definition.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """
    cls_uuid: str = "objectgoal"

    def __init__(
        self,
        sim,
        config: "DictConfig",
        dataset: "ObjectNavDatasetV1",
        *args: Any,
        **kwargs: Any,
    ):
        self._sim = sim
        self._dataset = dataset
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (1,)
        max_value = self.config.goal_spec_max_val - 1
        if self.config.goal_spec == "TASK_CATEGORY_ID":
            max_value = max(
                self._dataset.category_to_task_category_id.values()
            )

        return spaces.Box(
            low=0, high=max_value, shape=sensor_shape, dtype=np.int64
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: ObjectGoalNavEpisode,
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
        if self.config.goal_spec == "TASK_CATEGORY_ID":
            return np.array(
                [self._dataset.category_to_task_category_id[category_name]],
                dtype=np.int64,
            )
        elif self.config.goal_spec == "OBJECT_ID":
            obj_goal = episode.goals[0]
            assert isinstance(obj_goal, ObjectGoal)  # for type checking
            return np.array([obj_goal.object_name_id], dtype=np.int64)
        else:
            raise RuntimeError(
                "Wrong goal_spec specified for ObjectGoalSensor."
            )

@registry.register_sensor(name="ThoughtSensor")
class ThoughtSensor(Sensor):
    cls_uuid: str = "instruction"

    def __init__(self, sim, config: "DictConfig", *args: Any, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(512,), dtype=np.float32)

    def get_observation(
        self,
        observations: Dict[str, Observations],
        episode: ObjectGoalNavThinkingEpisode,
        *args: Any,
        **kwargs: Any
    ):
        # Return a float32 array to be compatible with PyTorch
        # TODO: Use task.thought if available from ThinkAction
        return np.random.normal(size=(512,)).astype(np.float32)

@registry.register_task(name="ObjectNavThinking-v1")
class ObjectNavigationThinkingTask(ObjectNavigationTask):
    r"""An Object Navigation Task class for a task specific methods.
    Used to explicitly state a type of the task in config.
    """
    def __init__(
        self,
        config: "DictConfig",
        sim: Simulator,
        dataset: Optional[Dataset] = None,
    ) -> None:
        super().__init__(config=config, sim=sim, dataset=dataset)
        self.thought: Optional[np.ndarray] = None
        self.last_image: Optional[np.ndarray] = None

    def reset(self, episode):
        observations = super().reset(episode)
        self.thought = None
        return observations

    def step(self, action: Dict[str, Any], episode: Episode):
        observation = super().step(action, episode)
        self.last_image = observation["rgb"]
        return observation


@registry.register_task_action
class ThinkAction(SimulatorTaskAction):
    name: str = "think"

    def reset(self, task: ObjectNavigationThinkingTask, *args: Any, **kwargs: Any):
        task.thought = None

    def step(self, task: ObjectNavigationThinkingTask, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        # TODO: Call VLM here to generate thought from task.last_image
        # For now, just generate random thought embedding
        task.thought = np.random.normal(size=(512,))
