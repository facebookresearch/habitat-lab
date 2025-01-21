#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np

from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.tasks.rearrange.rearrange_sensors import (
    BaseToObjectDistance,
    EEPositionSensor,
    EndEffectorToObjectDistance,
    EndEffectorToRestDistance,
    ForceTerminate,
    RearrangeReward,
    RobotForce,
)
from habitat.tasks.rearrange.utils import (
    get_camera_lookat_relative_to_vertical_line,
    rearrange_logger,
)


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
class RearrangePickReward(RearrangeReward):
    cls_uuid: str = "pick_reward"

    def __init__(self, *args, sim, config, task, **kwargs):
        self.cur_dist = -1.0
        self._prev_picked = False
        self._metric = None

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
            ],
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
        ee_to_object_distance = task.measurements.measures[
            EndEffectorToObjectDistance.cls_uuid
        ].get_metric()
        ee_to_rest_distance = task.measurements.measures[
            EndEffectorToRestDistance.cls_uuid
        ].get_metric()

        snapped_id = self._sim.grasp_mgr.snap_idx
        cur_picked = snapped_id is not None

        if cur_picked:
            dist_to_goal = ee_to_rest_distance
        else:
            dist_to_goal = ee_to_object_distance[str(task.targ_idx)]

        abs_targ_obj_idx = self._sim.scene_obj_ids[task.abs_targ_idx]

        did_pick = cur_picked and (not self._prev_picked)
        if did_pick:
            if snapped_id == abs_targ_obj_idx:
                self._metric += self._config.pick_reward
                # If we just transitioned to the next stage our current
                # distance is stale.
                self.cur_dist = -1
            else:
                # picked the wrong object
                self._metric -= self._config.wrong_pick_pen
                if self._config.wrong_pick_should_end:
                    rearrange_logger.debug(
                        "Grasped wrong object, ending episode."
                    )
                    self._task.should_end = True
                self._prev_picked = cur_picked
                self.cur_dist = -1
                return

        if self._config.use_diff:
            if self.cur_dist < 0:
                dist_diff = 0.0
            else:
                dist_diff = self.cur_dist - dist_to_goal

            # Filter out the small fluctuations
            dist_diff = round(dist_diff, 3)
            self._metric += self._config.dist_reward * dist_diff
        else:
            self._metric -= self._config.dist_reward * dist_to_goal
        self.cur_dist = dist_to_goal

        if not cur_picked and self._prev_picked:
            # Dropped the object
            self._metric -= self._config.drop_pen
            if self._config.drop_obj_should_end:
                self._task.should_end = True
            self._prev_picked = cur_picked
            self.cur_dist = -1
            return

        if self._config.max_target_distance != -1:
            # Robot is too far away from the target
            base_to_object_distance = task.measurements.measures[
                BaseToObjectDistance.cls_uuid
            ].get_metric()
            if (
                base_to_object_distance is not None
                and base_to_object_distance[str(task.targ_idx)]
                > self._config.max_target_distance
            ):
                self._task.should_end = True
                self._metric -= self._config.max_target_distance_pen
                return

        if self._config.non_desire_ee_local_pos_dis != -1:
            # Robot moves the arm to non-desire location
            assert (
                self._config.non_desire_ee_local_pos is not None
            ), "Please provide non_desire_ee_local_pos given non_desire_ee_local_pos_dis is non-negative"
            ee_local_pos = observations[EEPositionSensor.cls_uuid]
            distance = np.linalg.norm(
                np.array(ee_local_pos)
                - np.array(self._config.non_desire_ee_local_pos)
            )
            if distance < self._config.non_desire_ee_local_pos_dis:
                # The robot's EE is too closed to the non-desire ee pos
                self._task.should_end = True
                self._metric -= self._config.non_desire_ee_local_pos_pen
                return

        if self._config.camera_looking_down_angle != -1:
            # Get angle
            angle = get_camera_lookat_relative_to_vertical_line(
                self._sim.articulated_agent
            )
            # Get the bbox keys
            get_bbox_keys = [k for k in observations if "bbox" in k]
            # Check if there is target obejct in frame
            is_there_an_target_in_bbox = True
            if len(get_bbox_keys) != 0:
                is_there_an_target_in_bbox = (
                    np.sum(observations[get_bbox_keys[0]]) > 0
                )
            if (
                angle < self._config.camera_looking_down_angle
                and not is_there_an_target_in_bbox
            ):
                # The robot is looking down too much when there is no object in the frame
                self._task.should_end = True
                self._metric -= self._config.camera_looking_down_pen
                return

        self._prev_picked = cur_picked


@registry.register_measure
class RearrangePickSuccess(Measure):
    cls_uuid: str = "pick_success"

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
        abs_targ_obj_idx = self._sim.scene_obj_ids[task.abs_targ_idx]

        # Check that we are holding the right object and the object is actually
        # being held.
        self._metric = (
            abs_targ_obj_idx == self._sim.grasp_mgr.snap_idx
            and not self._sim.grasp_mgr.is_violating_hold_constraint()
            and (
                ee_to_rest_distance < self._config.ee_resting_success_threshold
                or self._config.ee_resting_success_threshold == -1.0
            )
        )
        self._prev_ee_pos = observations["ee_pos"]
