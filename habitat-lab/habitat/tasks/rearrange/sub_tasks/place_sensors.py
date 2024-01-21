#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import magnum as mn
import numpy as np

import habitat.sims.habitat_simulator.sim_utilities as sutils
import habitat_sim
from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.datasets.rearrange.samplers.receptacle import find_receptacles
from habitat.tasks.rearrange.rearrange_sensors import (
    EndEffectorToGoalDistance,
    EndEffectorToRestDistance,
    ForceTerminate,
    ObjAtGoal,
    ObjectToGoalDistance,
    RearrangeReward,
    RobotForce,
)
from habitat.tasks.rearrange.utils import rearrange_logger


@registry.register_measure
class ObjAtReceptacle(Measure):
    """
    Returns 1 if the agent has called the stop action and 0 otherwise.
    """

    cls_uuid: str = "obj_at_receptacle"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)
        self._targ_obj = []
        self._support_object_ids = []

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ObjAtReceptacle.cls_uuid

    def reset_metric(self, *args, task, **kwargs):
        # Cache the receptacle IDs for checking if the object is at a receptacle
        # Get the current target object
        idxs, _ = self._sim.get_targets()
        self._targ_obj = [
            self._sim.get_rigid_object_manager().get_object_by_id(
                self._sim.scene_obj_ids[idx]
            )
            for idx in idxs
        ]
        # Cache the receptacle IDs
        self._get_recepticle_ids()
        self.update_metric(*args, **kwargs)

    def _get_recepticle_ids(self):
        """Returns the list of receptacle IDs"""
        receptacles = find_receptacles(self._sim)
        self._support_object_ids = []
        for receptacle in receptacles:
            if receptacle.is_parent_object_articulated:
                ao_instance = self._sim.get_articulated_object_manager().get_object_by_handle(
                    receptacle.parent_object_handle
                )
                for (
                    object_id,
                    link_ix,
                ) in ao_instance.link_object_ids.items():
                    if receptacle.parent_link == link_ix:
                        self._support_object_ids += [
                            object_id,
                            ao_instance.object_id,
                        ]
                        break
            elif receptacle.parent_object_handle is not None:
                self._support_object_ids += [
                    self._sim.get_rigid_object_manager()
                    .get_object_by_handle(receptacle.parent_object_handle)
                    .object_id
                ]

    def update_metric(self, *args, **kwargs):
        idxs, goal_pos = self._sim.get_targets()
        scene_pos = self._sim.get_scene_pos()
        cur_obj_pos = scene_pos[idxs]

        # Check 1: Compute the height difference
        vertical_diff = np.linalg.norm(
            cur_obj_pos[:, [1]] - goal_pos[:, [1]], ord=2, axis=-1
        )

        # Check 2: place x, y location to the goal location
        horizontal_diff = np.linalg.norm(
            cur_obj_pos[:, [0, 2]] - goal_pos[:, [0, 2]], ord=2, axis=-1
        )

        # Check 3: Get the first hit object's height, and check if that height
        # is similar to the goal location's height
        surface_vertical_diff = []
        for i in range(vertical_diff.shape[0]):
            # Cast a ray to see if the object is on the receptacle
            ray = habitat_sim.geo.Ray()
            # Only support one object at a time
            ray.origin = mn.Vector3(cur_obj_pos[i])
            # Cast a ray from top to bottom
            ray.direction = mn.Vector3(0, -1.0, 0)
            raycast_results = self._sim.cast_ray(ray)
            if raycast_results.has_hits():
                surface_vertical_diff.append(
                    abs(raycast_results.hits[0].point[1] - goal_pos[i, 1])
                )
            else:
                surface_vertical_diff.append(-1.0)

        # Check 4: Use snap down function to check if the object can be placed
        snap_down_height_diff = []
        for ori_obj in self._targ_obj:
            # Note that this does not consider the height difference
            ori_obj_pos = ori_obj.translation
            snap_success_temp = False
            snap_success_temp = sutils.snap_down(
                self._sim, ori_obj, self._support_object_ids
            )
            # Check the height difference between the object and the receptacle
            if snap_success_temp:
                height_diff = ori_obj_pos[1] - ori_obj.translation[1]
                snap_down_height_diff.append(height_diff)
            else:
                snap_down_height_diff.append(-float("inf"))

        # Get the metric
        self._metric = {
            str(i): (
                vertical_diff[i] < self._config.vertical_diff_threshold
                or self._config.vertical_diff_threshold == -1
            )
            and (
                (
                    surface_vertical_diff[i] != -1
                    and surface_vertical_diff[i]
                    < self._config.surface_vertical_diff_threshold
                )
                or self._config.surface_vertical_diff_threshold == -1
            )
            and (
                horizontal_diff[i] < self._config.horizontal_diff_threshold
                or self._config.horizontal_diff_threshold == -1
            )
            and (
                (
                    snap_down_height_diff[i]
                    < self._config.snap_down_surface_vertical_diff_threshold
                    and snap_down_height_diff[i] >= 0.0
                )
                or self._config.snap_down_surface_vertical_diff_threshold == -1
            )
            for i in range(vertical_diff.shape[0])
        }


@registry.register_measure
class PlaceReward(RearrangeReward):
    cls_uuid: str = "place_reward"

    def __init__(self, *args, sim, config, task, **kwargs):
        self._prev_dist = -1.0
        self._prev_dropped = False
        self._metric = None

        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return PlaceReward.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        measures = [
            ObjectToGoalDistance.cls_uuid,
            ObjAtGoal.cls_uuid,
            EndEffectorToRestDistance.cls_uuid,
            RobotForce.cls_uuid,
            ForceTerminate.cls_uuid,
        ]
        if self._config.obj_at_receptacle_success:
            measures += [ObjAtReceptacle.cls_uuid]

        task.measurements.check_measure_dependencies(self.uuid, measures)
        self._prev_dist = -1.0
        self._prev_dropped = not self._sim.grasp_mgr.is_grasped

        super().reset_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        super().update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs
        )
        reward = self._metric

        ee_to_goal_dist = task.measurements.measures[
            EndEffectorToGoalDistance.cls_uuid
        ].get_metric()
        obj_to_goal_dist = task.measurements.measures[
            ObjectToGoalDistance.cls_uuid
        ].get_metric()
        ee_to_rest_distance = task.measurements.measures[
            EndEffectorToRestDistance.cls_uuid
        ].get_metric()
        obj_at_goal = task.measurements.measures[
            ObjAtGoal.cls_uuid
        ].get_metric()[str(task.targ_idx)]

        obj_at_receptacle = False
        if self._config.obj_at_receptacle_success:
            obj_at_receptacle = task.measurements.measures[
                ObjAtReceptacle.cls_uuid
            ].get_metric()[str(task.targ_idx)]

        snapped_id = self._sim.grasp_mgr.snap_idx
        cur_picked = snapped_id is not None

        if (not obj_at_goal) or cur_picked:
            if self._config.use_ee_dist:
                dist_to_goal = ee_to_goal_dist[str(task.targ_idx)]
            else:
                dist_to_goal = obj_to_goal_dist[str(task.targ_idx)]
            min_dist = self._config.min_dist_to_goal
        else:
            dist_to_goal = ee_to_rest_distance
            min_dist = 0.0

        if (not self._prev_dropped) and (not cur_picked):
            self._prev_dropped = True
            if (
                obj_at_goal and not self._config.obj_at_receptacle_success
            ) or (
                obj_at_receptacle and self._config.obj_at_receptacle_success
            ):
                reward += self._config.place_reward
                # If we just transitioned to the next stage our current
                # distance is stale.
                self._prev_dist = -1
            else:
                # Dropped at wrong location
                reward -= self._config.drop_pen
                if self._config.wrong_drop_should_end:
                    rearrange_logger.debug(
                        "Dropped to wrong place, ending episode."
                    )
                    self._task.should_end = True
                    self._metric = reward
                    return

        if dist_to_goal >= min_dist:
            if self._config.use_diff:
                if self._prev_dist < 0:
                    dist_diff = 0.0
                else:
                    dist_diff = self._prev_dist - dist_to_goal

                # Filter out the small fluctuations
                dist_diff = round(dist_diff, 3)
                reward += self._config.dist_reward * dist_diff
            else:
                reward -= self._config.dist_reward * dist_to_goal
        self._prev_dist = dist_to_goal

        self._metric = reward


@registry.register_measure
class PlaceSuccess(Measure):
    cls_uuid: str = "place_success"

    def __init__(self, sim, config, *args, **kwargs):
        self._config = config
        self._sim = sim
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return PlaceSuccess.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        measures = [
            ObjAtGoal.cls_uuid,
            EndEffectorToRestDistance.cls_uuid,
        ]
        if self._config.obj_at_receptacle_success:
            measures += [ObjAtReceptacle.cls_uuid]

        task.measurements.check_measure_dependencies(self.uuid, measures)
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        is_obj_at_goal = task.measurements.measures[
            ObjAtGoal.cls_uuid
        ].get_metric()[str(task.targ_idx)]

        obj_at_receptacle = False
        if self._config.obj_at_receptacle_success:
            obj_at_receptacle = task.measurements.measures[
                ObjAtReceptacle.cls_uuid
            ].get_metric()[str(task.targ_idx)]

        is_holding = self._sim.grasp_mgr.is_grasped

        ee_to_rest_distance = task.measurements.measures[
            EndEffectorToRestDistance.cls_uuid
        ].get_metric()

        self._metric = (
            not is_holding
            and (
                (is_obj_at_goal and not self._config.obj_at_receptacle_success)
                or (
                    obj_at_receptacle
                    and self._config.obj_at_receptacle_success
                )
            )
            and (
                ee_to_rest_distance < self._config.ee_resting_success_threshold
                or self._config.ee_resting_success_threshold == -1.0
            )
        )
