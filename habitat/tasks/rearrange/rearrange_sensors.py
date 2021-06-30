#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
import numpy as np
from gym import spaces

from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.nav.nav import PointGoalSensor
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.utils import get_angle

# TODO: @maksymets should be accessed through Robot API
EE_GRIPPER_OFFSET = mn.Vector3(0.08, 0, 0)


@registry.register_sensor
class TargetPointGoalGPSAndCompassSensor(PointGoalSensor):
    cls_uuid: str = "target_point_goal_gps_and_compass_sensor"

    def __init__(self, *args, task, **kwargs):
        self._sim: RearrangeSim
        self._task = task
        super().__init__(*args, task=task, **kwargs)

    def get_observation(self, observations, episode, *args, **kwargs):
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        rotation_world_agent = agent_state.rotation

        target_position = self._sim.get_target_objs_start()[0]
        return self._compute_pointgoal(
            agent_position, rotation_world_agent, target_position
        )


class MultiObjSensor(PointGoalSensor):
    def __init__(self, *args, task, **kwargs):
        self._task = task
        self._sim: RearrangeSim
        super(MultiObjSensor, self).__init__(*args, task=task, **kwargs)

    def _get_observation_space(self, *args, **kwargs):
        n_targets = self._task.get_n_targets()
        return spaces.Box(
            shape=(n_targets, 3),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )


@registry.register_sensor
class AbsTargetObjectSensor(MultiObjSensor):
    """
    This is the ground truth position
    """

    cls_uuid: str = "abs_obj_cur_sensor"

    def get_observation(self, observations, episode, *args, **kwargs):
        self._sim: RearrangeSim
        idxs, _ = self._sim.get_targets()
        scene_pos = self._sim.get_scene_pos()
        pos = scene_pos[idxs]

        return pos


@registry.register_sensor
class TargetStartSensor(MultiObjSensor):
    """
    Relative position from end effector to target object
    """

    cls_uuid: str = "obj_start_sensor"

    def get_observation(self, *args, observations, episode, **kwargs):
        self._sim: RearrangeSim
        global_T = self._sim.robot.ee_transform
        T_inv = global_T.inverted()
        pos = self._sim.get_target_objs_start()
        for i in range(pos.shape[0]):
            pos[i] = T_inv.transform_point(pos[i])

        return pos[self._task.targ_idx]


@registry.register_sensor
class AbsTargetStartSensor(MultiObjSensor):
    """
    Relative position from end effector to target object
    """

    cls_uuid: str = "abs_obj_start_sensor"

    def _get_observation_space(self, *args, **kwargs):
        n_targets = self._task.get_n_targets()
        return spaces.Box(
            shape=(n_targets, 3),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        pos = self._sim.get_target_objs_start()
        return pos


@registry.register_sensor
class GoalSensor(MultiObjSensor):
    """
    Relative to the end effector
    """

    cls_uuid: str = "obj_goal_sensor"

    def get_observation(self, observations, episode, *args, **kwargs):
        global_T = self._sim.robot.ee_transform
        T_inv = global_T.inverted()

        _, pos = self._sim.get_targets()
        for i in range(pos.shape[0]):
            pos[i] = T_inv.transform_point(pos[i])
        return pos


@registry.register_sensor
class AbsGoalSensor(MultiObjSensor):
    cls_uuid: str = "abs_obj_goal_sensor"

    def get_observation(self, *args, observations, episode, **kwargs):
        _, pos = self._sim.get_targets()
        return pos


@registry.register_sensor
class LocalizationSensor(Sensor):
    def __init__(self, *args, sim, config, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return "localization"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(4,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        trans = self._sim.robot.sim_obj.transformation
        forward = np.array([1.0, 0, 0])
        heading = np.array(trans.transform_vector(forward))
        forward = forward[[0, 2]]
        heading = heading[[0, 2]]

        heading_angle = get_angle(forward, heading)
        c = np.cross(forward, heading) < 0
        if not c:
            heading_angle = -1.0 * heading_angle
        return np.array([*trans.translation, heading_angle])


@registry.register_sensor
class JointSensor(Sensor):
    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return "joint"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(9,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        joints_pos = self._sim.robot.arm_joint_pos
        return np.array(joints_pos).astype(np.float32)


@registry.register_sensor
class TrackMarkerSensor(Sensor):
    """
    Will track the first marker from the simulator's track markers array
    relative to the robot's EE position
    """

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return "track_marker"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        pos = self._sim.get_track_markers_pos()[0]

        trans = self._sim.get_robot_transform()
        pos = trans.inverted().transform_point(pos)
        return np.array(pos).astype(np.float32)


@registry.register_sensor
class EeSensor(Sensor):
    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return "ee_pos"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        trans = self._sim.robot.sim_obj.transformation
        ee_pos = self._sim.robot.ee_transform.translation
        local_ee_pos = trans.inverted().transform_point(ee_pos)

        return np.array(local_ee_pos)


@registry.register_sensor
class IsHoldingSensor(Sensor):
    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return "is_holding"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(shape=(1,), low=0, high=1, dtype=np.float32)

    def get_observation(self, observations, episode, *args, **kwargs):
        snapped_id = self._sim.snapped_obj_id
        is_holding = snapped_id is not None

        return np.array(int(is_holding)).reshape((1,))


@registry.register_measure
class ObjectToGoalDistance(Measure):
    cls_uuid: str = "object_to_goal_distance"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ObjectToGoalDistance.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, episode, **kwargs):
        idxs, goal_pos = self._sim.get_targets()
        scene_pos = self._sim.get_scene_pos()
        target_pos = scene_pos[idxs]
        distances = np.linalg.norm(target_pos - goal_pos, ord=2, axis=-1)
        self._metric = {idx: dist for idx, dist in zip(idxs, distances)}


@registry.register_measure
class EndEffectorToObjectDistance(Measure):
    cls_uuid: str = "ee_to_object_distance"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return EndEffectorToObjectDistance.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, episode, **kwargs):
        ee_pos = self._sim.robot.ee_transform.translation

        idxs, _ = self._sim.get_targets()
        scene_pos = self._sim.get_scene_pos()
        target_pos = scene_pos[idxs]

        distances = np.linalg.norm(target_pos - ee_pos, ord=2, axis=-1)

        self._metric = {idx: dist for idx, dist in zip(idxs, distances)}


@registry.register_measure
class DummyMeasure(Measure):
    cls_uuid: str = "dummy_measure"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return DummyMeasure.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, episode, **kwargs):
        self._metric = 0


@registry.register_measure
class EndEffectorToPosDistance(Measure):
    cls_uuid: str = "ee_to_pos_distance"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        self._target_pos = np.zeros(3)
        super().__init__(**kwargs)

    def set_target_pos(self, target_pos):
        self._target_pos = target_pos

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return EndEffectorToPosDistance.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, episode, **kwargs):
        ee_pos = self._sim.robot.ee_transform.translation

        distance = np.linalg.norm(self._target_pos - ee_pos, ord=2)

        self._metric = distance


@registry.register_measure
class RearrangePickReward(Measure):
    cls_uuid: str = "rearrangepick_reward"

    def __init__(self, *args, sim, config, task, **kwargs):
        self._sim = sim
        self._config = config
        self._task = task
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return RearrangePickReward.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                EndEffectorToObjectDistance.cls_uuid,
                RearrangePickSuccess.cls_uuid,
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
        ee_to_object_distance = task.measurements.measures[
            EndEffectorToObjectDistance.cls_uuid
        ].get_metric()
        success = task.measurements.measures[
            RearrangePickSuccess.cls_uuid
        ].get_metric()
        reward = 0

        snapped_id = self._sim.snapped_obj_id
        cur_picked = snapped_id is not None
        ee_pos = observations["ee_pos"]

        if cur_picked:
            dist_to_goal = np.linalg.norm(ee_pos - task.desired_resting)
        else:
            dist_to_goal = ee_to_object_distance[task.targ_idx]

        abs_targ_obj_idx = self._sim.scene_obj_ids[task.abs_targ_idx]

        did_pick = cur_picked and (not self._task)
        if did_pick:
            if snapped_id == abs_targ_obj_idx:
                task.n_succ_picks += 1
                reward += self._config.PICK_REWARD
                # If we just transitioned to the next stage our current
                # distance is stale.
                self._task.cur_dist = -1
            else:
                # picked the wrong object
                reward -= self._config.WRONG_PICK_PEN
                if self._config.WRONG_PICK_SHOULD_END:
                    self._task.should_end = True
                self._metric = reward
                return

        if self._config.USE_DIFF:
            if self._task.cur_dist < 0:
                dist_diff = 0.0
            else:
                dist_diff = self._task.cur_dist - dist_to_goal

            # Filter out the small fluctuations
            dist_diff = round(dist_diff, 3)
            reward += self._config.DIST_REWARD * dist_diff
        else:
            reward -= self._config.DIST_REWARD * dist_to_goal
        self._task.cur_dist = dist_to_goal

        if not cur_picked and self._task.prev_picked:
            # Dropped the object
            reward -= self._config.DROP_PEN
            if self._config.DROP_OBJ_SHOULD_END:
                self._task.should_end = True
            self._metric = reward
            return

        if success:
            reward += self._config.SUCC_REWARD

        reward += self._get_coll_reward()

        if self._task._is_violating_hold_constraint():
            reward -= self._config.CONSTRAINT_VIOLATE_PEN

        if (
            self._task._config.FORCE_BASED
            and 0 < self._task.use_max_accum_force < self._task.accum_force
        ):
            reward -= self._config.FORCE_END_PEN

        self._task.prev_picked = cur_picked

        self._metric = reward

    def _get_coll_reward(self):
        reward = 0
        if self._task._config.FORCE_BASED:
            # Penalize the force that was added to the accumulated force at the
            # last time step.
            reward -= min(
                self._config.FORCE_PEN * self._task.add_force,
                self._config.MAX_FORCE_PEN,
            )
        else:
            delta_coll = self._task._delta_coll
            reward -= (
                self._config.ROBO_OBJ_COLL_PEN * delta_coll.robo_obj_colls
            )

            total_colls = (
                delta_coll.obj_scene_colls + delta_coll.robo_scene_colls
            )
            if self._task._config.COUNT_ROBO_OBJ_COLLS:
                total_colls += delta_coll.robo_obj_colls
            reward -= self._config.COLL_PEN * (min(1, total_colls))
        return reward


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
        self._prev_ee_pos = observations["ee_pos"]
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        ee_to_object_distance = task.measurements.measures[
            EndEffectorToObjectDistance.cls_uuid
        ].get_metric()
        self._metric = False
        # Is the agent holding the object and it's at the start?
        abs_targ_obj_idx = self._sim.scene_obj_ids[task.abs_targ_idx]
        obj_to_ee = ee_to_object_distance[task.targ_idx]

        # Check that we are holding the right object and the object is actually
        # being held.
        if (
            abs_targ_obj_idx == self._sim.snapped_obj_id
            and obj_to_ee < self._config.HOLD_THRESH
        ):
            rest_dist = np.linalg.norm(
                self._prev_ee_pos - task.desired_resting
            )
            if rest_dist < self._config.SUCC_THRESH:
                self._metric = True

        self._prev_ee_pos = observations["ee_pos"]
