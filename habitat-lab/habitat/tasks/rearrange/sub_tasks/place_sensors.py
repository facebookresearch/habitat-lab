#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
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
class ObjAnywhereOnGoal(Measure):
    cls_uuid: str = "obj_anywhere_on_goal"

    def __init__(self, sim, config, *args, **kwargs):
        self._config = config
        self._sim = sim
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ObjAnywhereOnGoal.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        self._sim.perform_discrete_collision_detection()
        cps = self._sim.get_physics_contact_points()

        abs_obj_id = self._sim.scene_obj_ids[task.abs_targ_idx]
        for cp in cps:
            if cp.object_id_a == abs_obj_id or cp.object_id_b == abs_obj_id:
                if cp.contact_distance < -0.01:
                    self._metric = False
                else:
                    other_obj_id = cp.object_id_a + cp.object_id_b - abs_obj_id
                    # Check if the other object has an id that is acceptable
                    self._metric = (
                        other_obj_id in self._sim.valid_goal_rec_obj_ids
                    )
                    # Additional check for receptacles that are not on a separate object
                    if self._metric and other_obj_id == -1:
                        contact_point = (
                            cp.position_on_a_in_ws
                            if other_obj_id == cp.object_id_a
                            else cp.position_on_b_in_ws
                        )
                        for n, r in self._sim.receptacles.items():
                            if r.check_if_point_on_surface(
                                self._sim, contact_point
                            ):
                                self._metric = (
                                    n in self._sim.valid_goal_rec_names
                                )
                                break

        self._metric = False


@registry.register_measure
class PlaceReward(RearrangeReward):
    cls_uuid: str = "place_reward"

    def __init__(self, *args, sim, config, task, **kwargs):
        self._prev_dist = -1.0
        self._prev_dropped = False
        self._metric = None
        self._use_ee_dist = config.use_ee_dist
        self._min_dist_to_goal = config.min_dist_to_goal
        self._place_reward = config.place_reward
        self._drop_pen = config.drop_pen
        self._wrong_drop_should_end = config.wrong_drop_should_end
        self._use_diff = config.use_diff
        self._dist_reward = config.dist_reward
        self._sparse_reward = config.sparse_reward
        self._place_anywhere = config.place_anywhere
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return PlaceReward.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                EndEffectorToRestDistance.cls_uuid,
                RobotForce.cls_uuid,
                ForceTerminate.cls_uuid,
            ],
        )
        if not self._sparse_reward:
            task.measurements.check_measure_dependencies(
                self.uuid,
                [
                    ObjectToGoalDistance.cls_uuid,
                    EndEffectorToGoalDistance.cls_uuid,
                ],
            )
        if self._place_anywhere:
            task.measurements.check_measure_dependencies(
                self.uuid,
                [
                    ObjAnywhereOnGoal.cls_uuid,
                ],
            )
        else:
            task.measurements.check_measure_dependencies(
                self.uuid,
                [
                    ObjAtGoal.cls_uuid,
                ],
            )

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

        ee_to_rest_distance = task.measurements.measures[
            EndEffectorToRestDistance.cls_uuid
        ].get_metric()

        if self._place_anywhere:
            obj_at_goal = task.measurements.measures[
                ObjAnywhereOnGoal.cls_uuid
            ].get_metric()
        else:
            obj_at_goal = task.measurements.measures[
                ObjAtGoal.cls_uuid
            ].get_metric()[str(task.abs_targ_idx)]

        snapped_id = self._sim.grasp_mgr.snap_idx
        cur_picked = snapped_id is not None

        if (not obj_at_goal) or cur_picked:
            if self._sparse_reward:
                dist_to_goal = 0.0
            elif self._use_ee_dist:
                ee_to_goal_dist = task.measurements.measures[
                    EndEffectorToGoalDistance.cls_uuid
                ].get_metric()
                dist_to_goal = ee_to_goal_dist[str(task.abs_targ_idx)]
            else:
                obj_to_goal_dist = task.measurements.measures[
                    ObjectToGoalDistance.cls_uuid
                ].get_metric()
                dist_to_goal = obj_to_goal_dist[str(task.abs_targ_idx)]
            min_dist = self._min_dist_to_goal
        else:
            dist_to_goal = ee_to_rest_distance
            min_dist = 0.0

        if (not self._prev_dropped) and (not cur_picked):
            self._prev_dropped = True
            if obj_at_goal:
                reward += self._place_reward
                # If we just transitioned to the next stage our current
                # distance is stale.
                self._prev_dist = -1
            else:
                # Dropped at wrong location
                reward -= self._drop_pen
                if self._wrong_drop_should_end:
                    rearrange_logger.debug(
                        "Dropped to wrong place, ending episode."
                    )
                    self._task.should_end = True
                    self._metric = reward
                    return

        if dist_to_goal >= min_dist:
            if self._use_diff:
                if self._prev_dist < 0:
                    dist_diff = 0.0
                else:
                    dist_diff = self._prev_dist - dist_to_goal

                # Filter out the small fluctuations
                dist_diff = round(dist_diff, 3)
                reward += self._dist_reward * dist_diff
            else:
                reward -= self._dist_reward * dist_to_goal
        self._prev_dist = dist_to_goal

        self._metric = reward


@registry.register_measure
class PlaceSuccess(Measure):
    cls_uuid: str = "place_success"

    def __init__(self, sim, config, *args, **kwargs):
        self._config = config
        self._ee_resting_success_threshold = (
            self._config.ee_resting_success_threshold
        )
        self._place_anywhere = self._config.place_anywhere
        self._sim = sim
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return PlaceSuccess.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                EndEffectorToRestDistance.cls_uuid,
            ],
        )
        if self._place_anywhere:
            task.measurements.check_measure_dependencies(
                self.uuid,
                [
                    ObjAnywhereOnGoal.cls_uuid,
                ],
            )
        else:
            task.measurements.check_measure_dependencies(
                self.uuid,
                [
                    ObjAtGoal.cls_uuid,
                ],
            )
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        if self._place_anywhere:
            is_obj_at_goal = task.measurements.measures[
                ObjAnywhereOnGoal.cls_uuid
            ].get_metric()
        else:
            is_obj_at_goal = task.measurements.measures[
                ObjAtGoal.cls_uuid
            ].get_metric()[str(task.abs_targ_idx)]
        is_holding = self._sim.grasp_mgr.is_grasped

        ee_to_rest_distance = task.measurements.measures[
            EndEffectorToRestDistance.cls_uuid
        ].get_metric()

        self._metric = (
            not is_holding
            and is_obj_at_goal
            and ee_to_rest_distance < self._ee_resting_success_threshold
        )
