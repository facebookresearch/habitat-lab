#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from gym import spaces

import habitat_sim
from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.rearrange.multi_task.rearrange_pddl import (
    RearrangeObjectTypes,
)
from habitat.tasks.rearrange.rearrange_sensors import RearrangeReward
from habitat.tasks.utils import cartesian_to_polar

BASE_ACTION_NAME = "BASE_VELOCITY"


@registry.register_sensor
class TargetOrGoalStartPointGoalSensor(Sensor):
    """
    GPS and compass sensor relative to the starting object position or goal
    position.
    """

    cls_uuid: str = "object_to_agent_gps_compass"

    def __init__(self, *args, sim, task, **kwargs):
        self._task = task
        self._sim = sim
        super().__init__(*args, task=task, **kwargs)

    def _get_uuid(self, *args, **kwargs):
        return TargetOrGoalStartPointGoalSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(2,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, task, *args, **kwargs):
        robot_T = self._sim.robot.base_transformation

        if task.nav_to_obj_type == RearrangeObjectTypes.GOAL_POSITION:
            to_pos = self._sim.get_targets()[1][self._task.targ_idx]
        elif task.nav_to_obj_type == RearrangeObjectTypes.RIGID_OBJECT:
            to_pos = self._sim.get_target_objs_start()[self._task.targ_idx]
        else:
            raise ValueError(
                f"Got navigate to object type {RearrangeObjectTypes.RIGID_OBJECT}"
            )

        dir_vector = robot_T.inverted().transform_point(to_pos)
        rho, phi = cartesian_to_polar(dir_vector[0], dir_vector[1])

        return np.array([rho, -phi], dtype=np.float32)


@registry.register_sensor
class NavToSkillSensor(Sensor):
    cls_uuid: str = "nav_to_skill"

    def __init__(self, sim, config, *args, **kwargs):
        self._config = config
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return NavToSkillSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(self._config.NUM_SKILLS,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, task, *args, **kwargs):
        ret = np.zeros(self._config.NUM_SKILLS, dtype=np.float32)
        if task.nav_to_task_name is None or task.domain is None:
            return ret
        skills = task.domain.action_names

        cur_idx = skills.index(task.nav_to_task_name)
        ret[cur_idx] = 1.0
        return ret


@registry.register_sensor
class DistToNavGoalSensor(Sensor):
    cls_uuid: str = "dist_to_nav_goal"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return DistToNavGoalSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(1,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, task, *args, **kwargs):
        agent_pos = self._sim.safe_snap_point(self._sim.robot.base_pos)
        distance_to_target = self._sim.geodesic_distance(
            agent_pos,
            task.nav_target_pos,
        )
        return np.array([distance_to_target])


@registry.register_sensor
class NavGoalSensor(Sensor):
    cls_uuid: str = "nav_goal"

    def _get_uuid(self, *args, **kwargs):
        return NavGoalSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, task, *args, **kwargs):
        return task.nav_target_pos.astype(np.float32)


@registry.register_sensor
class NavRotToGoalSensor(Sensor):
    """
    Warning: This represents priviledged information in the task.
    """

    cls_uuid: str = "nav_rot_to_goal_sensor"

    def _get_uuid(self, *args, **kwargs):
        return NavRotToGoalSensor.cls_uuid

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(1,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, task, *args, **kwargs):
        heading_angle = float(self._sim.robot.base_rot)
        angle_dist = np.arctan2(
            np.sin(heading_angle - task.nav_target_angle),
            np.cos(heading_angle - task.nav_target_angle),
        )
        return np.abs(angle_dist)


@registry.register_sensor
class OracleNavigationActionSensor(Sensor):
    cls_uuid: str = "oracle_nav_actions"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return OracleNavigationActionSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def _path_to_point(self, point):
        agent_pos = self._sim.robot.base_pos

        path = habitat_sim.ShortestPath()
        path.requested_start = agent_pos
        path.requested_end = point
        found_path = self._sim.pathfinder.find_path(path)
        if not found_path:
            return [agent_pos, point]
        return path.points

    def get_observation(self, task, *args, **kwargs):
        path = self._path_to_point(task.nav_target_pos)
        return path[1]


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
            **kwargs,
        )

    def _get_agent_pos(self):
        current_pos = self._sim.robot.base_pos
        return self._sim.safe_snap_point(current_pos)

    def _get_cur_geo_dist(self, task):
        distance_to_target = self._sim.geodesic_distance(
            self._get_agent_pos(),
            task.nav_target_pos,
        )

        if distance_to_target == np.inf:
            distance_to_target = self._prev_dist
        if distance_to_target is None:
            distance_to_target = 30
        return distance_to_target


@registry.register_measure
class NavToObjReward(RearrangeReward):
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
        self._give_turn_reward = False
        self._prev_dist = -1.0
        super().reset_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        reward = 0.0
        cur_dist = task.measurements.measures[DistToGoal.cls_uuid].get_metric()
        if self._prev_dist < 0.0:
            dist_diff = 0.0
        else:
            dist_diff = self._prev_dist - cur_dist

        reward += self._config.DIST_REWARD * dist_diff
        self._prev_dist = cur_dist

        bad_terminate_pen = task.measurements.measures[
            BadCalledTerminate.cls_uuid
        ].reward_pen
        reward -= bad_terminate_pen

        if (
            self._config.SHOULD_REWARD_TURN
            and cur_dist < self._config.TURN_REWARD_DIST
        ):
            self._give_turn_reward = True

            angle_dist = task.measurements.measures[
                RotDistToGoal.cls_uuid
            ].get_metric()

            if self._cur_angle_dist < 0:
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
            **kwargs,
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
            **kwargs,
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
                self.reward_pen -= self._config.BAD_TERM_PEN * (
                    remaining / self._config.ENVIRONMENT.MAX_EPISODE_STEPS
                )
            else:
                self.reward_pen = self._config.BAD_TERM_PEN
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

        super().reset_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
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
        self._end_on_stop = task.actions[BASE_ACTION_NAME].end_on_stop

        super().reset_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
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

        if self._config.MUST_CALL_STOP:
            if called_stop:
                if self._end_on_stop:
                    task.should_end = True
            else:
                self._metric = False

    def does_action_want_stop(self, task, obs):
        if self._config.HEURISTIC_STOP:
            angle_succ = (
                self._get_angle_dist(obs) < self._config.SUCCESS_ANGLE_DIST
            )
            obj_dist = np.linalg.norm(obs["dyn_obj_start_or_goal_sensor"])
            return angle_succ and (obj_dist < 1.0)

        return task.actions[BASE_ACTION_NAME].does_want_terminate
