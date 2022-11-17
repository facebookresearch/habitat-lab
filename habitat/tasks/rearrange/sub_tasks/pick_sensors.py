#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.tasks.rearrange.rearrange_sensors import (
    EndEffectorToObjectDistance,
    EndEffectorToRestDistance,
    ForceTerminate,
    RearrangeReward,
    RobotForce,
)
from habitat.tasks.rearrange.multi_task.composite_sensors import (
    DoesWantTerminate,
)
from habitat.tasks.nav.nav import DistanceToGoal
from habitat.tasks.rearrange.utils import rearrange_logger
from habitat.datasets.rearrange.rearrange_dataset import ObjectRearrangeEpisode
import numpy as np


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
class PickObjectExistsMeasure(Measure):
    cls_uuid: str = "pick_object_exists"

    def __init__(self, sim, config, task, *args, **kwargs):
        self._task = task
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return PickObjectExistsMeasure.cls_uuid

    def reset_metric(self, *args, **kwargs):
        self.update_metric(*args, **kwargs)

    def update_metric(self, *args, **kwargs):
        self._metric = int(self._task.pick_object_exists)


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
    cls_uuid: str = "rearrangepick_reward"

    def __init__(self, *args, sim, config, task, **kwargs):
        self.cur_dist = -1.0
        self._prev_picked = False
        self._metric = None
        self._config = config
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return RearrangePickReward.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                EndEffectorToObjectDistance.cls_uuid,
                RobotForce.cls_uuid,
                ForceTerminate.cls_uuid,
                RearrangePickSuccess.cls_uuid,
            ],
        )
        if self._config.ANY_INSTANCE:
            task.measurements.check_measure_dependencies(
                self.uuid, [DistanceToGoal.cls_uuid]
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

    def update_metric(self, *args, episode, task, observations, **kwargs):
        super().update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )
        if self._config.ANY_INSTANCE:
            ee_to_object_distance = task.measurements.measures[
                DistanceToGoal.cls_uuid
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
            if self._config.ANY_INSTANCE:
                permissible_obj_ids = [
                    self._sim.scene_obj_ids[g.object_id]
                    for g in episode.candidate_objects
                ]
            else:
                abs_targ_obj_idx = self._sim.scene_obj_ids[task.abs_targ_idx]
                permissible_obj_ids = [abs_targ_obj_idx]
            if snapped_id in permissible_obj_ids:
                self._metric += self._config.PICK_REWARD
                # If we just transitioned to the next stage our current
                # distance is stale.
                self.cur_dist = -1
            else:
                # picked the wrong object
                self._metric -= self._config.WRONG_PICK_PEN
                if self._config.WRONG_PICK_SHOULD_END:
                    rearrange_logger.debug(
                        "Grasped wrong object, ending episode."
                    )
                    self._task.should_end = True
                self._prev_picked = cur_picked
                self.cur_dist = -1
                return

        # Only reward the object to end effector distance changes to SPARSE, dense reward for bringing arm back to resting position
        if not self._config.SPARSE_REWARD or did_pick:
            if self._config.USE_DIFF:
                if self.cur_dist < 0:
                    dist_diff = 0.0
                else:
                    dist_diff = self.cur_dist - dist_to_goal

                # Filter out the small fluctuations
                dist_diff = round(dist_diff, 3)
                self._metric += self._config.DIST_REWARD * dist_diff
            else:
                self._metric -= self._config.DIST_REWARD * dist_to_goal
        self.cur_dist = dist_to_goal

        if not cur_picked and self._prev_picked:
            # Dropped the object
            self._metric -= self._config.DROP_PEN
            if self._config.DROP_OBJ_SHOULD_END:
                self._task.should_end = True
            self._prev_picked = cur_picked
            self.cur_dist = -1
            return
        self._prev_picked = cur_picked


@registry.register_measure
class PickBadCalledTerminate(Measure):
    cls_uuid: str = "pick_bad_called_terminate"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return PickBadCalledTerminate.cls_uuid

    def reset_metric(self, *args, task, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [DoesWantTerminate.cls_uuid, RearrangePickSuccess.cls_uuid],
        )
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        does_action_want_stop = task.measurements.measures[
            DoesWantTerminate.cls_uuid
        ].get_metric()
        is_succ = task.measurements.measures[
            RearrangePickSuccess.cls_uuid
        ].get_metric()

        self._metric = (not is_succ) and does_action_want_stop


@registry.register_measure
class RearrangePickSuccess(Measure):
    cls_uuid: str = "rearrangepick_success"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        self._prev_ee_pos = None
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return RearrangePickSuccess.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [EndEffectorToObjectDistance.cls_uuid]
        )
        if self._config.MUST_CALL_STOP:
            task.measurements.check_measure_dependencies(
                self.uuid, [DoesWantTerminate.cls_uuid]
            )
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

        if self._config.ANY_INSTANCE:
            permissible_obj_ids = [
                self._sim.scene_obj_ids[g.object_id]
                for g in episode.candidate_objects
            ]
        else:
            abs_targ_obj_idx = self._sim.scene_obj_ids[task.abs_targ_idx]
            permissible_obj_ids = [abs_targ_obj_idx]

        # Check that we are holding the right object and the object is actually
        # being held.
        if task.pick_object_exists:
            self._metric = self._sim.grasp_mgr.snap_idx is None
        else:
            self._metric = (
                self._sim.grasp_mgr.snap_idx in permissible_obj_ids
                and not self._sim.grasp_mgr.is_violating_hold_constraint()
            )
        self._metric = (
            self._metric
            and ee_to_rest_distance < self._config.EE_RESTING_SUCCESS_THRESHOLD
        )
        if self._config.MUST_CALL_STOP:
            does_action_want_stop = task.measurements.measures[
                DoesWantTerminate.cls_uuid
            ].get_metric()
            self._metric = self._metric and does_action_want_stop

        self._prev_ee_pos = observations["ee_pos"]
