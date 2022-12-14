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
        rom = self._sim.get_rigid_object_manager()
        aom = self._sim.get_articulated_object_manager()
        # import pdb; pdb.set_trace()
        self._sim.perform_discrete_collision_detection()
        cps = self._sim.get_physics_contact_points()
        relevant_cps = []

        idxs, _ = self._sim.get_targets()
        abs_obj_id = self._sim.scene_obj_ids[task.abs_targ_idx]
        for cp in cps:
            if cp.object_id_a == abs_obj_id or cp.object_id_b == abs_obj_id:

                relevant_cps.append(cp)
                if cp.contact_distance < -0.01:
                    self._metric = False
                else:
                    other_obj_id = cp.object_id_a + cp.object_id_b - abs_obj_id

                    def get_category_name(handle):
                        import re

                        return re.sub(
                            r"_[0-9]+",
                            "",
                            handle.split(":")[0][:-1].replace(
                                "frl_apartment_", ""
                            ),
                        )

                    if aom.get_library_has_id(other_obj_id):
                        handle = aom.get_object_handle_by_id(other_obj_id)
                    elif rom.get_library_has_id(other_obj_id):
                        handle = rom.get_object_handle_by_id(other_obj_id)
                    else:
                        self._metric = False
                        return

                    category = get_category_name(handle)
                    if category == self._sim.target_categories["goal_recep"]:
                        self._metric = True
                    else:
                        self._metric = False
                    return

        self._metric = False


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
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                EndEffectorToRestDistance.cls_uuid,
                RobotForce.cls_uuid,
                ForceTerminate.cls_uuid,
            ],
        )
        if not self._config.sparse_reward:
            task.measurements.check_measure_dependencies(
                self.uuid,
                [
                    ObjectToGoalDistance.cls_uuid,
                    EndEffectorToGoalDistance.cls_uuid,
                ],
            )
        if self._config.place_anywhere:
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

        if self._config.place_anywhere:
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
            if self._config.sparse_reward:
                dist_to_goal = 0.0
            elif self._config.use_ee_dist:
                ee_to_goal_dist = task.measurements.measures[
                    EndEffectorToGoalDistance.cls_uuid
                ].get_metric()
                dist_to_goal = ee_to_goal_dist[str(task.abs_targ_idx)]
            else:
                obj_to_goal_dist = task.measurements.measures[
                    ObjectToGoalDistance.cls_uuid
                ].get_metric()
                dist_to_goal = obj_to_goal_dist[str(task.abs_targ_idx)]
            min_dist = self._config.min_dist_to_goal
        else:
            dist_to_goal = ee_to_rest_distance
            min_dist = 0.0

        if (not self._prev_dropped) and (not cur_picked):
            self._prev_dropped = True
            if obj_at_goal:
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
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                EndEffectorToRestDistance.cls_uuid,
            ],
        )
        if self._config.place_anywhere:
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
        if self._config.place_anywhere:
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
            and ee_to_rest_distance < self._config.ee_resting_success_threshold
        )
