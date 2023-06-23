#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import numpy as np
from gym import spaces

from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.ovmm.utils import find_closest_goal_index_within_distance
from habitat.tasks.ovmm.sub_tasks.nav_to_obj_task import (
    OVMMDynNavRLEnv,
)
from habitat.tasks.rearrange.sub_tasks.nav_to_obj_sensors import (
    NavToObjReward,
    NavToObjSuccess,
    RotDistToGoal,
)
from habitat.tasks.utils import compute_pixel_coverage


@registry.register_sensor
class OVMMNavGoalSegmentationSensor(Sensor):
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
        self._blank_out_prob = self._config.blank_out_prob
        self.resolution = (
            sim.agents[0]
            ._sensors[self.panoptic_uuid]
            .specification()
            .resolution
        )
        self._num_channels = 2 if self._is_nav_to_obj else 1
        self._resolution = (
            sim.agents[0]
            ._sensors[self.panoptic_uuid]
            .specification()
            .resolution
        )
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
            # Skip if receptacle is not in the agent's viewport or if the instance
            # is selected to be blanked out randomly
            if (
                instance_id > max_obs_val
                or np.random.random() < self._blank_out_prob
            ):
                continue
            obs[pan_obs == instance_id] = 1
        return obs

    def get_observation(
        self, observations, *args, episode, task: OVMMDynNavRLEnv, **kwargs
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
        self._blank_out_prob = self._config.blank_out_prob
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
        self, observations, *args, episode, task: OVMMDynNavRLEnv, **kwargs
    ):
        obs = np.copy(observations[self.panoptic_uuid])
        obj_id_map = np.zeros(np.max(obs) + 1, dtype=np.int32)
        assert (
            task.loaded_receptacle_categories
        ), "Empty receptacle semantic IDs, task didn't cache them."
        for obj_id, semantic_id in task.receptacle_semantic_ids.items():
            instance_id = obj_id + self._instance_ids_start
            # Skip if receptacle is not in the agent's viewport or if the instance
            # is selected to be blanked out randomly
            if (
                instance_id >= obj_id_map.shape[0]
                or np.random.random() < self._blank_out_prob
            ):
                continue
            obj_id_map[instance_id] = semantic_id
        obs = obj_id_map[obs]
        return obs


@registry.register_measure
class OVMMRotDistToGoal(RotDistToGoal):
    """
    Computes angle between the agent's heading direction and the direction from agent to object. Selects the object with the closest viewpoint for computing this angle.
    """

    cls_uuid: str = "ovmm_rot_dist_to_goal"

    def __init__(self, *args, sim, config, dataset, task, **kwargs):
        self._is_nav_to_obj = task.is_nav_to_obj
        super().__init__(*args, sim=sim, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OVMMRotDistToGoal.cls_uuid

    def _get_targ(self, task, episode):
        if self._is_nav_to_obj:
            goals = episode.candidate_objects
        else:
            goals = episode.candidate_goal_receps
        closest_idx = find_closest_goal_index_within_distance(
            self._sim, goals, episode.episode_id, max_dist=-1
        )
        return goals[closest_idx].position


@registry.register_measure
class OVMMNavToObjSucc(NavToObjSuccess):
    """Whether the agent has navigated within `success_distance` of the center of the closest candidate goal object and oriented itself within `success_angle` of the receptacle

    Used for training nav skills used in OVMM baseline"""

    cls_uuid: str = "ovmm_nav_to_obj_success"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OVMMNavToObjSucc.cls_uuid

    @property
    def _rot_dist_to_goal_cls_uuid(self):
        return OVMMRotDistToGoal.cls_uuid

    def __init__(self, *args, sim, config, dataset, task, **kwargs):
        super().__init__(
            *args, sim=sim, config=config, dataset=dataset, task=task, **kwargs
        )
        self._min_object_coverage_iou = config.min_object_coverage_iou

    @property
    def _target_iou_coverage_cls_uuid(self):
        return TargetIoUCoverage.cls_uuid

    def reset_metric(self, *args, task, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [self._target_iou_coverage_cls_uuid],
        )
        super().reset_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        super().update_metric(*args, task=task, **kwargs)
        if self._must_look_at_targ and self._min_object_coverage_iou > 0:
            place_goal_iou = task.measurements.measures[
                self._target_iou_coverage_cls_uuid
            ].get_metric()
            self._metric = (
                self._metric
                and place_goal_iou >= self._min_object_coverage_iou
            )


@registry.register_measure
class OVMMNavToObjReward(NavToObjReward):
    cls_uuid: str = "ovmm_nav_to_obj_reward"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OVMMNavToObjReward.cls_uuid

    @property
    def _nav_to_obj_succ_cls_uuid(self):
        return OVMMNavToObjSucc.cls_uuid

    @property
    def _rot_dist_to_goal_cls_uuid(self):
        return OVMMRotDistToGoal.cls_uuid


@registry.register_measure
class TargetIoUCoverage(Measure):
    cls_uuid: str = "target_iou_coverage"

    def __init__(self, *args, sim, task, config, **kwargs):
        self._instance_ids_start = sim.habitat_config.instance_ids_start
        self._is_nav_to_obj = task.is_nav_to_obj
        self._max_goal_dist = config.max_goal_dist
        self._sim = sim
        super().__init__(*args, sim=sim, task=task, config=config, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return TargetIoUCoverage.cls_uuid

    def _get_object_id(self, goal):
        if self._is_nav_to_obj:
            return self._sim.scene_obj_ids[int(goal.object_id)]
        else:
            return int(goal.object_id)

    def _get_goals(self, episode):
        """key to access the goal in the episode"""
        if self._is_nav_to_obj:
            return episode.candidate_objects
        else:
            return episode.candidate_goal_receps

    def _filter_out_goals_not_in_view(self, goals, observations):
        """filters out goals that are not in the agent's viewport"""
        filtered_goals = []
        for goal in goals:
            instance_id = self._get_object_id(goal) + self._instance_ids_start
            if instance_id in observations["robot_head_panoptic"]:
                filtered_goals.append(goal)
        return filtered_goals

    def _get_closest_object_id_in_view(self, episode, observations):
        """returns the object id of the closest object to the agent"""
        goals = self._get_goals(episode)
        goals = self._filter_out_goals_not_in_view(goals, observations)
        if len(goals) == 0:
            return -1
        closest_idx = find_closest_goal_index_within_distance(
            self._sim,
            goals,
            episode.episode_id,
            max_dist=self._max_goal_dist,
            use_all_viewpoints=True,
        )
        if closest_idx == -1:
            return -1
        return self._get_object_id(goals[closest_idx])

    def reset_metric(self, *args, task, **kwargs):
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, episode, task, observations, **kwargs):
        object_id = self._get_closest_object_id_in_view(episode, observations)
        if object_id == -1:
            self._metric = 0
            return
        self._metric = compute_pixel_coverage(
            observations["robot_head_panoptic"],
            object_id + self._instance_ids_start,
        )
