#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from habitat.core.embodied_task import Measure
from habitat.core.registry import registry

BASE_ACTION_NAME = "BASE_VELOCITY"


class GeoMeasure(Measure):
    def __init__(self, *args, sim, config, task, **kwargs):
        self._config = config
        self._sim = sim
        self._prev_dist = None
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self._prev_dist = self._get_cur_geo_dist(task)
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs
        )

    def _get_agent_pos(self):
        current_pos = self._sim.robot.base_pos
        return self._sim.pathfinder.snap_point(current_pos)

    def _get_cur_geo_dist(self, task):
        distance_to_target = self._sim.geodesic_distance(
            self._get_agent_pos(),
            task.nav_target_pos,
        )

        if distance_to_target == np.inf:
            distance_to_target = self._prev_dist
            print("Distance is infinity", "returning ", distance_to_target)
        return distance_to_target


@registry.register_measure
class NavToObjReward(GeoMeasure):
    cls_uuid: str = "nav_to_obj_reward"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return NavToObjReward.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                NavToObjSuccess.cls_uuid,
                BadCalledTerminate.cls_uuid,
                DistToGoal.cls_uuid,
                RotDistToGoal.cls_uuid,
            ],
        )
        self._cur_angle_dist = -1.0
        super().reset_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        reward = self._config.SLACK_REWARD
        cur_dist = self._get_cur_geo_dist(task)
        cur_dist = task.measurements.measures[DistToGoal.cls_uuid].get_metric()
        cur_angle_dist = task.measurements.measures[
            RotDistToGoal.cls_uuid
        ].get_metric()

        reward += self._prev_dist - cur_dist
        self._prev_dist = cur_dist

        success = task.measurements.measures[
            NavToObjSuccess.cls_uuid
        ].get_metric()

        bad_terminate_pen = task.measurements.measures[
            BadCalledTerminate.cls_uuid
        ].reward_pen
        reward -= bad_terminate_pen

        if success:
            reward += self._config.SUCCESS_REWARD

        if (
            self._config.SHOULD_REWARD_TURN
            and cur_dist < self._config.TURN_REWARD_DIST
        ):

            angle_dist = task.measurements.measures[
                RotDistToGoal.cls_uuid
            ].get_metric()

            if cur_angle_dist < 0:
                angle_diff = 0.0
            else:
                angle_diff = self._cur_angle_dist - angle_dist

            reward += self._config.ANGLE_DIST_REWARD * angle_diff
            self._cur_angle_dist = angle_dist

        self._metric = reward


@registry.register_measure
class SPLToObj(GeoMeasure):
    cls_uuid: str = "spl_to_obj"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return SPLToObj.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self._start_dist = self._get_cur_geo_dist(task)
        self._previous_pos = self._get_agent_pos()
        super().reset_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        is_success = float(
            task.measurements.measures[NavToObjSuccess.cls_uuid].get_metric()
        )
        current_pos = self._get_agent_pos()
        dist = np.linalg.norm(current_pos - self._previous_pos)
        self._previous_pos = current_pos
        return is_success * (self._start_dist / max(self._start_dist, dist))


@registry.register_measure
class DistToGoal(GeoMeasure):
    cls_uuid: str = "dist_to_goal"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return DistToGoal.cls_uuid

    def update_metric(self, *args, episode, task, observations, **kwargs):
        self._metric = self._get_cur_geo_dist(task)


@registry.register_measure
class RotDistToGoal(GeoMeasure):
    cls_uuid: str = "rot_dist_to_goal"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return RotDistToGoal.cls_uuid

    def update_metric(self, *args, episode, task, observations, **kwargs):
        heading_angle = float(self._sim.robot.base_rot)
        angle_dist = np.arctan2(
            np.sin(heading_angle - task.nav_target_angle),
            np.cos(heading_angle - task.nav_target_angle),
        )
        self._metric = np.abs(angle_dist)


@registry.register_measure
class BadCalledTerminate(GeoMeasure):
    cls_uuid: str = "bad_called_terminate"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return BadCalledTerminate.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self.reward_pen = 0.0
        super().reset_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        success_measure = task.measurements.measures[NavToObjSuccess.cls_uuid]
        if (
            success_measure.does_action_want_stop(task, observations)
            and not success_measure.get_metric()
        ):
            if self._config.DECAY_BAD_TERM:
                remaining = (
                    self._config.ENVIRONMENT.MAX_EPISODE_STEPS - self._n_steps
                )
                self.reward_pen -= -self._config.BAD_TERM_PEN * (
                    remaining / self._config.ENVIRONMENT.MAX_EPISODE_STEPS
                )
            else:
                self.reward_pen = -self._config.BAD_TERM_PEN
            self._metric = 1.0
        else:
            self._metric = 0.0


@registry.register_measure
class NavToPosSucc(GeoMeasure):
    cls_uuid: str = "nav_to_pos_success"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return NavToPosSucc.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [DistToGoal.cls_uuid],
        )
        self._action_can_stop = task.actions[BASE_ACTION_NAME].end_on_stop

        super().reset_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        dist = task.measurements.measures[DistToGoal.cls_uuid].get_metric()
        self._metric = dist < self._config.SUCCESS_DISTANCE


@registry.register_measure
class NavToObjSuccess(GeoMeasure):
    cls_uuid: str = "nav_to_obj_success"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return NavToObjSuccess.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        # Get the end_on_stop property from the action
        task.measurements.check_measure_dependencies(
            self.uuid,
            [NavToPosSucc.cls_uuid, RotDistToGoal.cls_uuid],
        )
        self._action_can_stop = task.actions[BASE_ACTION_NAME].end_on_stop

        super().reset_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        angle_dist = task.measurements.measures[
            RotDistToGoal.cls_uuid
        ].get_metric()

        nav_pos_succ = task.measurements.measures[
            NavToPosSucc.cls_uuid
        ].get_metric()

        if self._config.MUST_LOOK_AT_TARG:
            self._metric = (
                nav_pos_succ and angle_dist < self._config.SUCCESS_ANGLE_DIST
            )
        else:
            self._metric = nav_pos_succ
        called_stop = self.does_action_want_stop(task, observations)
        if self._action_can_stop and not called_stop:
            self._metric = False

    def does_action_want_stop(self, task, obs):
        if self._config.HEURISTIC_STOP:
            angle_succ = (
                self._get_angle_dist(obs) < self._config.SUCCESS_ANGLE_DIST
            )
            obj_dist = np.linalg.norm(obs["dyn_obj_start_or_goal_sensor"])
            return angle_succ and (obj_dist < 1.0)

        return task.actions[BASE_ACTION_NAME].does_want_terminate
