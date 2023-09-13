# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import TYPE_CHECKING, Any, List, Optional

import attr
import numpy as np
from gym import spaces

import habitat_sim
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import RGBSensor, Sensor, SensorTypes
from habitat.tasks.nav.nav import NavigationEpisode, NavigationTask
from habitat.utils.geometry_utils import quaternion_from_coeff
from habitat_sim import bindings as hsim
from habitat_sim.agent.agent import AgentState, SixDOFPose

if TYPE_CHECKING:
    from omegaconf import DictConfig


@registry.register_sensor
class MultiGoalSensor(Sensor):
    r"""A sensor for GOAT goal specification.
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
    cls_uuid: str = "multigoal"

    def __init__(
        self,
        sim,
        config: "DictConfig",
        dataset: "GoatDatasetV1",
        *args: Any,
        **kwargs: Any,
    ):
        self._sim = sim
        self._dataset = dataset

        sensors = self._sim.sensor_suite.sensors
        rgb_sensor_uuids = [
            uuid
            for uuid, sensor in sensors.items()
            if isinstance(sensor, RGBSensor)
        ]
        if len(rgb_sensor_uuids) != 1:
            raise ValueError(
                f"ImageGoalNav requires one RGB sensor, {len(rgb_sensor_uuids)} detected"
            )

        (self._rgb_sensor_uuid,) = rgb_sensor_uuids
        self._current_episode_id = None
        self._current_image_goal = None

        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TEXT

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        max_tasks = 5
        max_landmarks = 20
        max_semantic_classes = 1000

        goal_dict = {
            "task_type": spaces.Text(min_length=0, max_length=30),
            "semantic_id": spaces.Box(low=0, high=max_semantic_classes),
            "target": spaces.Text(min_length=0, max_length=30),
            "landmarks": spaces.Tuple(
                [
                    spaces.Text(min_length=0, max_length=30)
                    for _ in range(max_landmarks)
                ]
            ),
            "image": self._sim.sensor_suite.observation_spaces.spaces[
                self._rgb_sensor_uuid
            ],
        }

        return spaces.Tuple([spaces.Dict(goal_dict) for _ in range(max_tasks)])

    def _add_sensor(self, img_params, sensor_uuid: str) -> None:
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = sensor_uuid
        spec.sensor_type = habitat_sim.SensorType.COLOR
        spec.resolution = img_params.image_dimensions
        spec.hfov = img_params.hfov
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        self._sim.add_sensor(spec)

        agent = self._sim.get_agent(0)
        agent_state = agent.get_state()
        agent.set_state(
            AgentState(
                position=agent_state.position,
                rotation=agent_state.rotation,
                sensor_states={
                    **agent_state.sensor_states,
                    sensor_uuid: SixDOFPose(
                        position=np.array(img_params.position),
                        rotation=quaternion_from_coeff(img_params.rotation),
                    ),
                },
            ),
            infer_sensor_states=False,
        )

    def _remove_sensor(self, sensor_uuid: str) -> None:
        agent = self._sim.get_agent(0)
        del self._sim._sensors[sensor_uuid]
        hsim.SensorFactory.delete_subtree_sensor(agent.scene_node, sensor_uuid)
        del agent._sensors[sensor_uuid]
        agent.agent_config.sensor_specifications = [
            s
            for s in agent.agent_config.sensor_specifications
            if s.uuid != sensor_uuid
        ]

    def get_image_goal(self, episode, goal_idx):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id} {goal_idx}"
        if episode_uniq_id == self._current_episode_id:
            return self._current_image_goal
        img_params = episode.goals[goal_idx][0].image_goals[
            episode.tasks[goal_idx]["goal_image_id"]
        ]

        sensor_uuid = f"{self.cls_uuid}_sensor"
        self._add_sensor(img_params, sensor_uuid)

        self._sim._sensors[sensor_uuid].draw_observation()
        self._current_image_goal = self._sim._sensors[
            sensor_uuid
        ].get_observation()[:, :, :3]

        self._remove_sensor(sensor_uuid)

        self._current_episode_id = episode_uniq_id
        return self._current_image_goal

    def get_observation(
        self,
        observations,
        *args: Any,
        episode,  # TODO
        **kwargs: Any,
    ) -> Optional[np.ndarray]:

        if len(episode.goals) == 0:
            logger.error(
                f"No goal specified for episode {episode.episode_id}."
            )
            return None

        # if not isinstance(episode.goals[0], ObjectGoal):
        #     logger.error(
        #         f"First goal should be ObjectGoal, episode {episode.episode_id}."
        #     )
        #     return None

        goals, vocabulary = [], []
        for goal_idx, task in enumerate(episode.tasks):
            goal = {
                "type": task["task_type"],
            }

            if task["task_type"] == "objectnav":
                goal["target"] = task["object_category"]
            elif task["task_type"] == "languagenav":
                if "llm_response" in task.keys():
                    target = task["llm_response"]["target"]
                    landmarks = task["llm_response"]["landmark"]
                    if target in landmarks:
                        landmarks.remove(target)

                    if "wall" in landmarks:
                        landmarks.remove("wall")  # unhelpful landmark

                    target = "_".join(target.split())
                    landmarks = [
                        "_".join(landmark.split()) for landmark in landmarks
                    ]
                    goal["landmarks"] = landmarks
                    goal["target"] = target
                else:
                    goal["target"] = task["object_category"]

            elif task["task_type"] == "imagenav":
                goal["target"] = task["object_category"]
                goal["image"] = self.get_image_goal(episode, goal_idx)

            if goal["target"] not in vocabulary:
                vocabulary.append(goal["target"])
            if "landmarks" in goal:
                if goal["landmarks"] not in vocabulary:
                    vocabulary += goal["landmarks"]

            goal["semantic_id"] = vocabulary.index(goal["target"]) + 1
            goals.append(goal)
        return goals


@registry.register_task(name="Goat-v1")
class GoatTask(NavigationTask):  # TODO
    r"""A GOAT Task class for a task specific methods.
    Used to explicitly state a type of the task in config.
    """


@attr.s(auto_attribs=True, kw_only=True)
class GoatEpisode(NavigationEpisode):
    r"""Goat Episode

    :param object_category: Category of the obect
    """
    object_category: Optional[str] = None
    tasks: List[NavigationEpisode] = []

    @property
    def goals_keys(self) -> str:
        r"""Dictionary of goals types and corresonding keys"""
        goals_keys = {ep["task_type"]: [] for ep in self.tasks}

        for ep in self.tasks:
            if ep["task_type"] == "objectnav":
                goal_key = f"{os.path.basename(self.scene_id)}_{ep['object_category']}"

            elif ep["task_type"] == "imagenav":
                sid = os.path.basename(self.scene_id)
                for x in [".glb", ".basis"]:
                    sid = sid[: -len(x)] if sid.endswith(x) else sid
                goal_key = f"{sid}_{ep['goal_object_id']}"

            elif ep["task_type"] == "languagenav":
                goal_key = f"{os.path.basename(self.scene_id)}_{ep['object_instance_id']}"

            goals_keys[ep["task_type"]].append(goal_key)

        return goals_keys

    def goals_keys_with_sequence(self) -> str:
        r"""The key to retrieve the goals"""
        goals_keys = []

        for ep in self.tasks:
            if ep["task_type"] == "objectnav":
                goal_key = f"{os.path.basename(self.scene_id)}_{ep['object_category']}"

            elif ep["task_type"] == "imagenav":
                sid = os.path.basename(self.scene_id)
                for x in [".glb", ".basis"]:
                    sid = sid[: -len(x)] if sid.endswith(x) else sid
                goal_key = f"{sid}_{ep['goal_object_id']}"

            elif ep["task_type"] == "languagenav":
                goal_key = f"{os.path.basename(self.scene_id)}_{ep['object_instance_id']}"

            goals_keys.append(goal_key)

        return goals_keys
