#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import numpy as np
from gym import spaces

import habitat_sim
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.rearrange.sub_tasks.cat_nav_to_obj_task import (
    CatDynNavRLEnv,
)
from habitat.tasks.rearrange.sub_tasks.nav_to_obj_sensors import (
    NavToObjReward,
    NavToObjSuccess,
    RotDistToGoal,
)


@registry.register_sensor
class OvmmNavGoalSegmentationSensor(Sensor):
    cls_uuid: str = "ovmm_nav_goal_segmentation"
    panoptic_uuid: str = "robot_head_panoptic"

    def __init__(
        self,
        sim,
        config,
        dataset,
        task,
        *args: Any,
        **kwargs: Any,
    ):
        self._config = config
        self._sim = sim
        self._instance_ids_start = self._sim.habitat_config.instance_ids_start
        self._is_nav_to_obj = task.is_nav_to_obj
        self.resolution = (
            sim.agents[0]
            ._sensors[self.panoptic_uuid]
            .specification()
            .resolution
        )
        self._num_channels = 2 if self._is_nav_to_obj else 1
        self._resolution = sim.agents[0]._sensors[self.panoptic_uuid].specification().resolution
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(
                self._resolution[0],
                self._resolution[1],
                self._num_channels,
            ),
            low=0,
            high=1,
            dtype=np.int32,
        )

    def _get_obs_channel(self, pan_obs, max_obs_val, goals, goals_type):
        pan_obs = pan_obs.squeeze(axis=-1)
        obs = np.zeros_like(pan_obs)
        for goal in goals:
            if goals_type == "obj":
                obj_id = self._sim.scene_obj_ids[int(goal.object_id)]
            elif goals_type == "rec":
                rom = self._sim.get_rigid_object_manager()
                handle = self._sim.receptacles[
                    goal.object_name
                ].parent_object_handle
                obj_id = rom.get_object_id_by_handle(handle)
            instance_id = obj_id + self._instance_ids_start
            # Skip if object is not in the agent's viewport
            if instance_id > max_obs_val:
                continue
            obs[pan_obs == instance_id] = 1
        return obs

    def get_observation(
        self, observations, *args, episode, task: CatDynNavRLEnv, **kwargs
    ):
        pan_obs = observations[self.panoptic_uuid]
        max_obs_val = np.max(pan_obs)
        obs = np.zeros(
            (self.resolution[0], self.resolution[1], self._num_channels),
            dtype=np.int32,
        )
        if self._is_nav_to_obj:
            obs[..., 0] = self._get_obs_channel(
                pan_obs,
                max_obs_val,
                episode.candidate_objects_hard,
                "obj",
            )
            obs[..., 1] = self._get_obs_channel(
                pan_obs,
                max_obs_val,
                episode.candidate_start_receps,
                "rec",
            )
        else:
            obs[..., 0] = self._get_obs_channel(
                pan_obs,
                max_obs_val,
                episode.candidate_goal_receps,
                "rec",
            )

        return obs


@registry.register_sensor
class ReceptacleSegmentationSensor(Sensor):
    cls_uuid: str = "receptacle_segmentation"
    panoptic_uuid: str = "robot_head_panoptic"

    def __init__(
        self,
        sim,
        config,
        *args: Any,
        **kwargs: Any,
    ):
        self._config = config
        self._sim = sim
        self._instance_ids_start = self._sim.habitat_config.instance_ids_start
        self.resolution = (
            sim.agents[0]
            ._sensors[self.panoptic_uuid]
            .specification()
            .resolution
        )
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(
                self.resolution[0],
                self.resolution[1],
                1,
            ),
            low=np.iinfo(np.uint32).min,
            high=np.iinfo(np.uint32).max,
            dtype=np.int32,
        )

    def get_observation(
        self, observations, *args, episode, task: CatDynNavRLEnv, **kwargs
    ):
        obs = np.copy(observations[self.panoptic_uuid])
        obj_id_map = np.zeros(np.max(obs) + 1, dtype=np.int32)
        ssert (
            task.loaded_receptacle_categories
        ), "Empty receptacle semantic IDs, task didn't cache them."
        for obj_id, semantic_id in task.receptacle_semantic_ids.items():
            instance_id = obj_id + self._instance_ids_start
            # Skip if receptacle is not in the agent's viewport
            if instance_id >= obj_id_map.shape[0]:
                continue
            obj_id_map[instance_id] = semantic_id
        obs = obj_id_map[obs]
        return obs


@registry.register_measure
class OvmmRotDistToGoal(RotDistToGoal):
    """
    Computes angle between the agent's heading direction and the direction from agent to object. Selects the object with the closest viewpoint for computing this angle.
    """

    cls_uuid: str = "ovmm_rot_dist_to_goal"

    def __init__(self, *args, sim, config, dataset, task, **kwargs):
        self._is_nav_to_obj = task.is_nav_to_obj
        super().__init__(*args, sim=sim, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OvmmRotDistToGoal.cls_uuid

    def _get_targ(self, task, episode):
        if self._is_nav_to_obj:
            goals = episode.candidate_objects
        else:
            goals = episode.candidate_goal_receps
        goal_pos = [g.position for g in goals]
        goal_view_points = [
            g.view_points[0].agent_state.position for g in goals
        ]
        path = habitat_sim.MultiGoalShortestPath()
        path.requested_start = self._sim.robot.base_pos
        path.requested_ends = goal_view_points
        self._sim.pathfinder.find_path(path)
        assert (
            path.closest_end_point_index != -1
        ), f"None of the goals are reachable from current position for episode {episode.episode_id}"
        # RotDist to closest goal
        targ = goal_pos[path.closest_end_point_index]
        return targ


@registry.register_measure
class OvmmNavToObjSucc(NavToObjSuccess):
    """Whether the agent has navigated within `success_distance` of the center of the closest candidate goal object and oriented itself within `success_angle` of the receptacle

    Used for training nav skills used in OVMM baseline"""

    cls_uuid: str = "ovmm_nav_to_obj_success"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OvmmNavToObjSucc.cls_uuid

    @property
    def _rot_dist_to_goal_cls_uuid(self):
        return OvmmRotDistToGoal.cls_uuid


@registry.register_measure
class OvmmNavToObjReward(NavToObjReward):
    cls_uuid: str = "ovmm_nav_to_obj_reward"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OvmmNavToObjReward.cls_uuid

    @property
    def _nav_to_obj_succ_cls_uuid(self):
        return OvmmNavToObjSucc.cls_uuid

    @property
    def _rot_dist_to_goal_cls_uuid(self):
        return OvmmRotDistToGoal.cls_uuid
