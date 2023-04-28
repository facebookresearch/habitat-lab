#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import magnum as mn
import numpy as np

from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.tasks.nav.nav import DistanceToGoal, DistanceToGoalReward
from habitat.tasks.rearrange.rearrange_sensors import (
    EndEffectorToObjectDistance,
    EndEffectorToRestDistance,
    ForceTerminate,
    RearrangeReward,
    RobotForce,
)
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.utils import (
    UsesRobotInterface,
    get_camera_transform,
    rearrange_logger,
)
from habitat.utils.geometry_utils import cosine


@registry.register_measure
class PickDistanceToGoal(DistanceToGoal, UsesRobotInterface, Measure):
    cls_uuid: str = "pick_distance_to_goal"

    def get_base_position(self):
        assert isinstance(self._sim, RearrangeSim)
        return self._sim.robot.base_pos

    def get_end_effector_position(self):
        assert isinstance(self._sim, RearrangeSim)
        return self._sim.get_robot_data(
            self.robot_id
        ).robot.ee_transform.translation


@registry.register_measure
class PickDistanceToGoalReward(
    DistanceToGoalReward, UsesRobotInterface, Measure
):
    cls_uuid: str = "pick_distance_to_goal_reward"

    @property
    def distance_to_goal_cls(self):
        return PickDistanceToGoal


@registry.register_measure
class DidPickObjectMeasure(Measure):
    cls_uuid: str = "did_pick_object"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return DidPickObjectMeasure.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self._did_pick = False
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, episode, **kwargs):
        self._did_pick = self._did_pick or self._sim.grasp_mgr.is_grasped
        self._metric = int(self._did_pick)


@registry.register_measure
class PickedObjectCategory(Measure):
    cls_uuid: str = "picked_object_category"

    def __init__(self, sim, config, dataset, *args, **kwargs):
        self._sim = sim
        self._dataset = dataset
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return PickedObjectCategory.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, episode, **kwargs):
        rom = self._sim.get_rigid_object_manager()
        # TODO: currently for ycb objects category extracted from handle name. Maintain and use a handle to category mapping
        if (
            self._sim.grasp_mgr.is_grasped
            and self._sim.grasp_mgr.snap_idx is not None
        ):
            self._metric = self._dataset.obj_category_to_obj_category_id[
                rom.get_object_handle_by_id(
                    self._sim.grasp_mgr.snap_idx
                ).split(":")[0][4:-1]
            ]
        else:
            self._metric = -1


@registry.register_measure
class RearrangePickReward(RearrangeReward):
    cls_uuid: str = "pick_reward"

    def __init__(self, *args, sim, config, task, **kwargs):
        self.cur_dist = -1.0
        self._prev_picked = False
        self._metric = None
        self._pick_reward = config.pick_reward
        self._wrong_pick_pen = config.wrong_pick_pen
        self._wrong_pick_should_end = config.wrong_pick_should_end
        self._use_diff = config.use_diff
        self._dist_reward = config.dist_reward
        self._drop_pen = config.drop_pen
        self._drop_obj_should_end = config.drop_obj_should_end
        self._object_goal = config.object_goal
        self._sparse_reward = config.sparse_reward
        self._angle_reward_min_dist = config.angle_reward_min_dist
        self._angle_reward_scale = config.angle_reward_scale
        self._task = task
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return RearrangePickReward.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                RobotForce.cls_uuid,
                ForceTerminate.cls_uuid,
            ],
        )
        if self._object_goal:
            task.measurements.check_measure_dependencies(
                self.uuid, [PickDistanceToGoal.cls_uuid]
            )
        else:
            task.measurements.check_measure_dependencies(
                self.uuid, [EndEffectorToObjectDistance.cls_uuid]
            )

        self.cur_dist = -1.0
        self._prev_picked = self._sim.grasp_mgr.snap_idx is not None

        super().reset_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def get_camera_angle_reward(self, obj_pos):
        """Calculates angle between gripper line-of-sight and given global position."""

        # Get the camera transformation
        cam_T = get_camera_transform(self._sim)
        # Get object location in camera frame
        cam_obj_pos = cam_T.inverted().transform_point(obj_pos).normalized()

        # Get angle between (normalized) location and the vector that the camera should
        # look at
        reward = cosine(cam_obj_pos, mn.Vector3(0, 1, 0))

        return reward

    def closest_goal_position(self, episode):
        # Find the goal that is closest based on l2-distance
        targets = np.array(
            [goal.position for goal in episode.candidate_objects]
        )
        closest_goal_index = np.argmin(
            np.linalg.norm(
                np.expand_dims(self._sim.robot.base_pos, 0) - targets, axis=1
            )
        )
        targ = targets[closest_goal_index]
        return targ

    def update_metric(self, *args, episode, task, observations, **kwargs):
        super().update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )
        if self._object_goal:
            ee_to_object_distance = task.measurements.measures[
                PickDistanceToGoal.cls_uuid
            ].get_metric()
        else:
            ee_to_object_distance = task.measurements.measures[
                EndEffectorToObjectDistance.cls_uuid
            ].get_metric()[str(task.abs_targ_idx)]
        ee_to_rest_distance = task.measurements.measures[
            EndEffectorToRestDistance.cls_uuid
        ].get_metric()

        snapped_id = self._sim.grasp_mgr.snap_idx
        cur_picked = snapped_id is not None

        if cur_picked:
            dist_to_goal = ee_to_rest_distance
        else:
            dist_to_goal = ee_to_object_distance

        did_pick = cur_picked and (not self._prev_picked)
        if did_pick:
            if self._object_goal:
                permissible_obj_ids = [
                    self._sim.scene_obj_ids[int(g.object_id)]
                    for g in episode.candidate_objects
                ]
            else:
                abs_targ_obj_idx = self._sim.scene_obj_ids[task.abs_targ_idx]
                permissible_obj_ids = [abs_targ_obj_idx]
            if snapped_id in permissible_obj_ids:
                self._metric += self._pick_reward
                # If we just transitioned to the next stage our current
                # distance is stale.
                self.cur_dist = -1
            else:
                # picked the wrong object
                self._metric -= self._wrong_pick_pen
                if self._wrong_pick_should_end:
                    rearrange_logger.debug(
                        "Grasped wrong object, ending episode."
                    )
                    self._task.should_end = True
                self._prev_picked = cur_picked
                self.cur_dist = -1
                return

        # If _sparse_reward is True, use dense reward only after the object gets picked for bringing arm to resting position
        if not self._sparse_reward or did_pick:
            if self._use_diff:
                if self.cur_dist < 0:
                    dist_diff = 0.0
                else:
                    dist_diff = self.cur_dist - dist_to_goal

                # Filter out the small fluctuations
                dist_diff = round(dist_diff, 3)
                self._metric += self._dist_reward * dist_diff
            else:
                self._metric -= self._dist_reward * dist_to_goal
            # Ensure no object is picked, cur_dist is not stale and that agent is within
            # self._angle_reward_min_dist of a goal
            if (
                not did_pick
                and self.cur_dist != -1
                and self.cur_dist < self._angle_reward_min_dist
            ):
                # closest based on l2 distance
                closest_pos = self.closest_goal_position(episode)
                self._metric += (
                    self._angle_reward_scale
                    * self.get_camera_angle_reward(closest_pos)
                )
        self.cur_dist = dist_to_goal

        if not cur_picked and self._prev_picked:
            # Dropped the object
            self._metric -= self._drop_pen
            if self._drop_obj_should_end:
                self._task.should_end = True
            self._prev_picked = cur_picked
            self.cur_dist = -1
            return
        self._prev_picked = cur_picked


@registry.register_measure
class RearrangePickSuccess(Measure):
    cls_uuid: str = "pick_success"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        self._ee_resting_success_threshold = (
            self._config.ee_resting_success_threshold
        )
        self._object_goal = self._config.object_goal
        self._prev_ee_pos = None
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return RearrangePickSuccess.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self._prev_ee_pos = observations["ee_pos"]

        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        ee_to_rest_distance = task.measurements.measures[
            EndEffectorToRestDistance.cls_uuid
        ].get_metric()

        # Is the agent holding the object and it's at the start?

        if self._object_goal:
            permissible_obj_ids = [
                self._sim.scene_obj_ids[int(g.object_id)]
                for g in episode.candidate_objects
            ]
        else:
            abs_targ_obj_idx = self._sim.scene_obj_ids[task.abs_targ_idx]
            permissible_obj_ids = [abs_targ_obj_idx]

        # Check that we are holding the right object and the object is actually
        # being held.
        self._metric = (
            self._sim.grasp_mgr.snap_idx in permissible_obj_ids
            and not self._sim.grasp_mgr.is_violating_hold_constraint()
            and ee_to_rest_distance < self._ee_resting_success_threshold
        )

        self._prev_ee_pos = observations["ee_pos"]
