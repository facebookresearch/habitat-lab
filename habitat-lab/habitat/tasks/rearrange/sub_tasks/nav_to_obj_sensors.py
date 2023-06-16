#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

import numpy as np
from gym import spaces

import habitat_sim
from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.nav.nav import EpisodicCompassSensor, EpisodicGPSSensor
from habitat.tasks.rearrange.rearrange_sensors import (
    DoesWantTerminate,
    RearrangeReward,
)
from habitat.tasks.rearrange.utils import UsesRobotInterface
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_from_coeff

if TYPE_CHECKING:
    from omegaconf import DictConfig

BASE_ACTION_NAME = "base_velocity"


@registry.register_sensor
class NavGoalPointGoalSensor(UsesRobotInterface, Sensor):
    """
    GPS and compass sensor relative to the starting object position or goal
    position.
    """

    cls_uuid: str = "goal_to_agent_gps_compass"

    def __init__(self, *args, sim, task, **kwargs):
        self._task = task
        self._sim = sim
        super().__init__(*args, task=task, **kwargs)

    def _get_uuid(self, *args, **kwargs):
        return NavGoalPointGoalSensor.cls_uuid

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
        robot_T = self._sim.get_robot_data(
            self.robot_id
        ).robot.base_transformation

        dir_vector = robot_T.inverted().transform_point(task.nav_goal_pos)
        rho, phi = cartesian_to_polar(dir_vector[0], dir_vector[1])

        return np.array([rho, -phi], dtype=np.float32)


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


@registry.register_measure
class NavToObjReward(RearrangeReward):
    cls_uuid: str = "nav_to_obj_reward"

    def __init__(self, *args, sim, config, task, **kwargs):
        self._dist_reward = config.dist_reward
        self._should_reward_turn = config.should_reward_turn
        self._turn_reward_dist = config.turn_reward_dist
        self._angle_dist_reward = config.angle_dist_reward
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return NavToObjReward.cls_uuid

    @property
    def _dist_to_goal_cls_uuid(self):
        return DistToGoal.cls_uuid

    @property
    def _nav_to_obj_succ_cls_uuid(self):
        return NavToObjSuccess.cls_uuid

    @property
    def _rot_dist_to_goal_cls_uuid(self):
        return RotDistToGoal.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                self._nav_to_obj_succ_cls_uuid,
                self._dist_to_goal_cls_uuid,
                self._rot_dist_to_goal_cls_uuid,
            ],
        )
        self._cur_angle_dist = -1.0
        self._prev_dist = -1.0
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
            **kwargs
        )
        reward = self._metric
        cur_dist = task.measurements.measures[
            self._dist_to_goal_cls_uuid
        ].get_metric()
        if self._prev_dist < 0.0:
            dist_diff = 0.0
        else:
            dist_diff = self._prev_dist - cur_dist

        reward += self._dist_reward * dist_diff
        self._prev_dist = cur_dist

        if self._should_reward_turn and cur_dist < self._turn_reward_dist:
            angle_dist = task.measurements.measures[
                self._rot_dist_to_goal_cls_uuid
            ].get_metric()

            if self._cur_angle_dist < 0:
                angle_diff = 0.0
            else:
                angle_diff = self._cur_angle_dist - angle_dist

            reward += self._angle_dist_reward * angle_diff
            self._cur_angle_dist = angle_dist

        self._metric = reward


@registry.register_measure
class DistToGoal(Measure):
    cls_uuid: str = "dist_to_goal"

    def __init__(self, *args, sim, config, task, **kwargs):
        self._config = config
        self._sim = sim
        self._prev_dist = None
        self._use_shortest_path_cache = config.use_shortest_path_cache
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self._prev_dist = self._get_cur_geo_dist(task, episode)
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def _get_goals(self, task, episode):
        if len(task.nav_goal_pos.shape) == 1:
            goals = np.expand_dims(task.nav_goal_pos, axis=0)
        else:
            goals = task.nav_goal_pos
        return goals

    def _get_cur_geo_dist(self, task, episode):
        goals = self._get_goals(task, episode)
        distance_to_target = self._sim.geodesic_distance(
            self._sim.robot.base_pos,
            goals,
            episode=episode if self._use_shortest_path_cache else None,
        )
        if distance_to_target == np.inf:
            distance_to_target = self._prev_dist
        if distance_to_target is None:
            distance_to_target = 30
        return distance_to_target

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return DistToGoal.cls_uuid

    def update_metric(self, *args, episode, task, observations, **kwargs):
        self._metric = self._get_cur_geo_dist(task, episode)


@registry.register_sensor(name="RobotStartGPSSensor")
class RobotStartGPSSensor(EpisodicGPSSensor):
    cls_uuid: str = "robot_start_gps"

    def __init__(self, sim, config: "DictConfig", *args, **kwargs):
        super().__init__(sim=sim, config=config)

    def get_agent_start_position(self, episode, task):
        return task._robot_start_position

    def get_agent_start_rotation(self, episode, task):
        return quaternion_from_coeff(
            task._robot_start_rotation
        )

    def get_agent_current_position(self, sim):
        return sim.robot.sim_obj.translation


@registry.register_sensor(name="RobotStartCompassSensor")
class RobotStartCompassSensor(EpisodicCompassSensor):
    cls_uuid: str = "robot_start_compass"

    def __init__(self, sim, config: "DictConfig", *args, **kwargs):
        super().__init__(sim=sim, config=config)

    def get_agent_start_rotation(self, episode, task):
        return quaternion_from_coeff(
            task._robot_start_rotation
        )

    def get_agent_current_rotation(self, sim):
        curr_quat = sim.robot.sim_obj.rotation
        curr_rotation = [
            curr_quat.vector.x,
            curr_quat.vector.y,
            curr_quat.vector.z,
            curr_quat.scalar,
        ]
        return quaternion_from_coeff(
            curr_rotation
        )


@registry.register_measure
class RotDistToGoal(Measure):
    cls_uuid: str = "rot_dist_to_goal"

    def __init__(self, *args, sim, **kwargs):
        self._sim = sim
        super().__init__(*args, sim=sim, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return RotDistToGoal.cls_uuid

    def reset_metric(self, *args, **kwargs):
        self.update_metric(
            *args,
            **kwargs,
        )

    def _get_targ(self, task, episode):
        if len(task.nav_goal_pos.shape) == 2:
            path = habitat_sim.MultiGoalShortestPath()
            path.requested_start = self._sim.robot.base_pos
            path.requested_ends = task.nav_goal_pos
            self._sim.pathfinder.find_path(path)
            assert (
                path.closest_end_point_index != -1
            ), f"None of the goals are reachable from current position for episode {episode.episode_id}"
            # RotDist to closest goal
            targ = task.nav_goal_pos[path.closest_end_point_index]
        else:
            targ = task.nav_goal_pos
        return targ

    def update_metric(self, *args, episode, task, observations, **kwargs):
        targ = self._get_targ(task, episode)

        robot = self._sim.robot

        # Get the base transformation
        T = robot.base_transformation
        # Do transformation
        pos = T.inverted().transform_point(targ)
        # Project to 2D plane (x,y,z=0)
        pos[2] = 0.0
        # Unit vector of the pos
        pos = pos.normalized()
        # Define the coordinate of the robot
        pos_robot = np.array([1.0, 0.0, 0.0])
        # Get the angle
        angle = np.arccos(np.dot(pos, pos_robot))
        self._metric = np.abs(float(angle))


@registry.register_measure
class NavToPosSucc(Measure):
    cls_uuid: str = "nav_to_pos_success"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return NavToPosSucc.cls_uuid

    def __init__(self, *args, config, **kwargs):
        self._config = config
        self._success_distance = self._config.success_distance
        super().__init__(*args, config=config, **kwargs)

    @property
    def _dist_to_goal_cls_uuid(self):
        return DistToGoal.cls_uuid

    def reset_metric(self, *args, task, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [self._dist_to_goal_cls_uuid],
        )
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, episode, task, observations, **kwargs):
        dist = task.measurements.measures[
            self._dist_to_goal_cls_uuid
        ].get_metric()
        self._metric = dist < self._success_distance


@registry.register_measure
class NavToObjSuccess(Measure):
    cls_uuid: str = "nav_to_obj_success"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return NavToObjSuccess.cls_uuid

    def reset_metric(self, *args, task, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [self._nav_to_pos_succ_cls_uuid, self._rot_dist_to_goal_cls_uuid],
        )
        self.update_metric(*args, task=task, **kwargs)

    def __init__(self, *args, config, **kwargs):
        self._config = config
        self._must_look_at_targ = self._config.must_look_at_targ
        self._success_angle_dist = self._config.success_angle_dist
        self._must_call_stop = self._config.must_call_stop
        super().__init__(*args, config=config, **kwargs)

    @property
    def _nav_to_pos_succ_cls_uuid(self):
        return NavToPosSucc.cls_uuid

    @property
    def _rot_dist_to_goal_cls_uuid(self):
        return RotDistToGoal.cls_uuid

    def update_metric(self, *args, episode, task, observations, **kwargs):
        angle_dist = task.measurements.measures[
            self._rot_dist_to_goal_cls_uuid
        ].get_metric()

        nav_pos_succ = task.measurements.measures[
            self._nav_to_pos_succ_cls_uuid
        ].get_metric()

        called_stop = task.measurements.measures[
            DoesWantTerminate.cls_uuid
        ].get_metric()

        if self._must_look_at_targ:
            self._metric = (
                nav_pos_succ and angle_dist < self._success_angle_dist
            )
        else:
            self._metric = nav_pos_succ

        if self._must_call_stop:
            if called_stop:
                task.should_end = True
            else:
                self._metric = False
