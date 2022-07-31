#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.tasks.rearrange.rearrange_sensors import (
    EndEffectorToRestDistance,
    ForceTerminate,
    ObjAtGoal,
    ObjectToGoalDistance,
    RearrangeReward,
    RobotForce,
)
from habitat.tasks.rearrange.utils import rearrange_logger


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
                ObjectToGoalDistance.cls_uuid,
                ObjAtGoal.cls_uuid,
                EndEffectorToRestDistance.cls_uuid,
                RobotForce.cls_uuid,
                ForceTerminate.cls_uuid,
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
        obj_to_goal_dist = task.measurements.measures[
            ObjectToGoalDistance.cls_uuid
        ].get_metric()
        ee_to_rest_distance = task.measurements.measures[
            EndEffectorToRestDistance.cls_uuid
        ].get_metric()
        obj_at_goal = task.measurements.measures[
            ObjAtGoal.cls_uuid
        ].get_metric()[str(task.abs_targ_idx)]

        snapped_id = self._sim.grasp_mgr.snap_idx
        cur_picked = snapped_id is not None

        if (not obj_at_goal) or cur_picked:
            dist_to_goal = obj_to_goal_dist[str(task.abs_targ_idx)]
        else:
            dist_to_goal = ee_to_rest_distance

        if (not self._prev_dropped) and (not cur_picked):
            self._prev_dropped = True
            if obj_at_goal:
                reward += self._config.PLACE_REWARD
                # If we just transitioned to the next stage our current
                # distance is stale.
                self._prev_dist = -1
            else:
                # Dropped at wrong location
                reward -= self._config.DROP_PEN
                if self._config.WRONG_DROP_SHOULD_END:
                    rearrange_logger.debug(
                        "Dropped to wrong place, ending episode."
                    )
                    self._task.should_end = True
                    self._metric = reward
                    return

        if self._config.USE_DIFF:
            if self._prev_dist < 0:
                dist_diff = 0.0
            else:
                dist_diff = self._prev_dist - dist_to_goal

            # Filter out the small fluctuations
            dist_diff = round(dist_diff, 3)
            reward += self._config.DIST_REWARD * dist_diff
        else:
            reward -= self._config.DIST_REWARD * dist_to_goal
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
                ObjAtGoal.cls_uuid,
                EndEffectorToRestDistance.cls_uuid,
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
            and ee_to_rest_distance < self._config.EE_RESTING_SUCCESS_THRESHOLD
        )
