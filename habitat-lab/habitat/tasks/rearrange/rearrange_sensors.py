#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from collections import defaultdict, deque

import numpy as np
from gym import spaces

from habitat.articulated_agents.humanoids import KinematicHumanoid
from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.nav.nav import PointGoalSensor
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.utils import (
    CollisionDetails,
    UsesArticulatedAgentInterface,
    batch_transform_point,
    get_angle_to_pos,
    get_camera_object_angle,
    get_camera_transform,
    rearrange_logger,
)
from habitat.tasks.utils import cartesian_to_polar


class MultiObjSensor(PointGoalSensor):
    """
    Abstract parent class for a sensor that specifies the locations of all targets.
    """

    def __init__(self, *args, task, **kwargs):
        self._task = task
        self._sim: RearrangeSim
        super().__init__(*args, task=task, **kwargs)

    def _get_observation_space(self, *args, **kwargs):
        n_targets = self._task.get_n_targets()
        return spaces.Box(
            shape=(n_targets * 3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )


@registry.register_sensor
class TargetCurrentSensor(UsesArticulatedAgentInterface, MultiObjSensor):
    """
    This is the ground truth object position sensor relative to the robot end-effector coordinate frame.
    """

    cls_uuid: str = "obj_goal_pos_sensor"

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        self._sim: RearrangeSim
        T_inv = (
            self._sim.get_agent_data(self.agent_id)
            .articulated_agent.ee_transform()
            .inverted()
        )

        idxs, _ = self._sim.get_targets()
        scene_pos = self._sim.get_scene_pos()
        pos = scene_pos[idxs]

        for i in range(pos.shape[0]):
            pos[i] = T_inv.transform_point(pos[i])

        return pos.reshape(-1)


@registry.register_sensor
class TargetStartSensor(UsesArticulatedAgentInterface, MultiObjSensor):
    """
    Relative position from end effector to target object
    """

    cls_uuid: str = "obj_start_sensor"

    def get_observation(self, *args, observations, episode, **kwargs):
        self._sim: RearrangeSim
        global_T = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.ee_transform()
        T_inv = global_T.inverted()
        pos = self._sim.get_target_objs_start()
        return batch_transform_point(pos, T_inv, np.float32).reshape(-1)


class PositionGpsCompassSensor(UsesArticulatedAgentInterface, Sensor):
    def __init__(self, *args, sim, task, **kwargs):
        self._task = task
        self._sim = sim
        super().__init__(*args, task=task, **kwargs)

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        n_targets = self._task.get_n_targets()
        self._polar_pos = np.zeros(n_targets * 2, dtype=np.float32)
        return spaces.Box(
            shape=(n_targets * 2,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def _get_positions(self) -> np.ndarray:
        raise NotImplementedError("Must override _get_positions")

    def get_observation(self, task, *args, **kwargs):
        pos = self._get_positions()
        articulated_agent_T = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.base_transformation

        rel_pos = batch_transform_point(
            pos, articulated_agent_T.inverted(), np.float32
        )

        for i, rel_obj_pos in enumerate(rel_pos):
            rho, phi = cartesian_to_polar(rel_obj_pos[0], rel_obj_pos[1])
            self._polar_pos[(i * 2) : (i * 2) + 2] = [rho, -phi]
        # TODO: This is a hack. For some reason _polar_pos in overridden by the other agent.
        return self._polar_pos.copy()


@registry.register_sensor
class TargetStartGpsCompassSensor(PositionGpsCompassSensor):
    cls_uuid: str = "obj_start_gps_compass"

    def _get_uuid(self, *args, **kwargs):
        return TargetStartGpsCompassSensor.cls_uuid

    def _get_positions(self) -> np.ndarray:
        return self._sim.get_target_objs_start()


@registry.register_sensor
class TargetGoalGpsCompassSensor(PositionGpsCompassSensor):
    cls_uuid: str = "obj_goal_gps_compass"

    def _get_uuid(self, *args, **kwargs):
        return TargetGoalGpsCompassSensor.cls_uuid

    def _get_positions(self) -> np.ndarray:
        _, pos = self._sim.get_targets()
        return pos


@registry.register_sensor
class AbsTargetStartSensor(MultiObjSensor):
    """
    Relative position from end effector to target object
    """

    cls_uuid: str = "abs_obj_start_sensor"

    def get_observation(self, observations, episode, *args, **kwargs):
        pos = self._sim.get_target_objs_start()
        return pos.reshape(-1)


@registry.register_sensor
class GoalSensor(UsesArticulatedAgentInterface, MultiObjSensor):
    """
    Relative to the end effector
    """

    cls_uuid: str = "obj_goal_sensor"

    def get_observation(self, observations, episode, *args, **kwargs):
        global_T = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.ee_transform()
        T_inv = global_T.inverted()

        _, pos = self._sim.get_targets()
        return batch_transform_point(pos, T_inv, np.float32).reshape(-1)


@registry.register_sensor
class AbsGoalSensor(MultiObjSensor):
    cls_uuid: str = "abs_obj_goal_sensor"

    def get_observation(self, *args, observations, episode, **kwargs):
        _, pos = self._sim.get_targets()
        return pos.reshape(-1)


@registry.register_sensor
class JointSensor(UsesArticulatedAgentInterface, Sensor):
    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim
        self._arm_joint_mask = config.arm_joint_mask

    def _get_uuid(self, *args, **kwargs):
        return "joint"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        if config.arm_joint_mask is not None:
            assert config.dimensionality == np.sum(config.arm_joint_mask)
        return spaces.Box(
            shape=(config.dimensionality,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def _get_mask_joint(self, joints_pos):
        """Select the joint location"""
        mask_joints_pos = []
        for i in range(len(self._arm_joint_mask)):
            if self._arm_joint_mask[i]:
                mask_joints_pos.append(joints_pos[i])
        return mask_joints_pos

    def get_observation(self, observations, episode, *args, **kwargs):
        joints_pos = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.arm_joint_pos
        if self._arm_joint_mask is not None:
            joints_pos = self._get_mask_joint(joints_pos)
        return np.array(joints_pos, dtype=np.float32)


@registry.register_sensor
class HumanoidJointSensor(UsesArticulatedAgentInterface, Sensor):
    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return "humanoid_joint_sensor"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(config.dimensionality,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        curr_agent = self._sim.get_agent_data(self.agent_id).articulated_agent
        if isinstance(curr_agent, KinematicHumanoid):
            joints_pos = curr_agent.get_joint_transform()[0]
            return np.array(joints_pos, dtype=np.float32)
        else:
            return np.zeros(self.observation_space.shape, dtype=np.float32)


@registry.register_sensor
class JointVelocitySensor(UsesArticulatedAgentInterface, Sensor):
    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return "joint_vel"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(config.dimensionality,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        joints_pos = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.arm_velocity
        return np.array(joints_pos, dtype=np.float32)


@registry.register_sensor
class EEPositionSensor(UsesArticulatedAgentInterface, Sensor):
    cls_uuid: str = "ee_pos"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return EEPositionSensor.cls_uuid

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
        trans = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.base_transformation
        ee_pos = (
            self._sim.get_agent_data(self.agent_id)
            .articulated_agent.ee_transform()
            .translation
        )
        local_ee_pos = trans.inverted().transform_point(ee_pos)

        return np.array(local_ee_pos, dtype=np.float32)


@registry.register_sensor
class RelativeRestingPositionSensor(UsesArticulatedAgentInterface, Sensor):
    cls_uuid: str = "relative_resting_position"

    def _get_uuid(self, *args, **kwargs):
        return RelativeRestingPositionSensor.cls_uuid

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, task, *args, **kwargs):
        base_trans = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.base_transformation
        ee_pos = (
            self._sim.get_agent_data(self.agent_id)
            .articulated_agent.ee_transform()
            .translation
        )
        local_ee_pos = base_trans.inverted().transform_point(ee_pos)

        relative_desired_resting = task.desired_resting - local_ee_pos

        return np.array(relative_desired_resting, dtype=np.float32)


@registry.register_sensor
class RestingPositionSensor(Sensor):
    """
    Desired resting position in the articulated_agent coordinate frame.
    """

    cls_uuid: str = "resting_position"

    def _get_uuid(self, *args, **kwargs):
        return RestingPositionSensor.cls_uuid

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, task, *args, **kwargs):
        return np.array(task.desired_resting, dtype=np.float32)


@registry.register_sensor
class LocalizationSensor(UsesArticulatedAgentInterface, Sensor):
    """
    The position and angle of the articulated_agent in world coordinates.
    """

    cls_uuid = "localization_sensor"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return LocalizationSensor.cls_uuid

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
        articulated_agent = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent
        T = articulated_agent.base_transformation
        forward = np.array([1.0, 0, 0])
        heading_angle = get_angle_to_pos(T.transform_vector(forward))
        return np.array(
            [*articulated_agent.base_pos, heading_angle], dtype=np.float32
        )


@registry.register_sensor
class IsHoldingSensor(UsesArticulatedAgentInterface, Sensor):
    """
    Binary if the robot is holding an object or grasped onto an articulated object.
    """

    cls_uuid: str = "is_holding"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return IsHoldingSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(shape=(1,), low=0, high=1, dtype=np.float32)

    def get_observation(self, observations, episode, *args, **kwargs):
        return np.array(
            int(self._sim.get_agent_data(self.agent_id).grasp_mgr.is_grasped),
            dtype=np.float32,
        ).reshape((1,))


@registry.register_measure
class ObjectToGoalDistance(Measure):
    """
    Euclidean distance from the target object to the goal.
    """

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
        self._metric = {str(idx): dist for idx, dist in enumerate(distances)}


@registry.register_measure
class GfxReplayMeasure(Measure):
    cls_uuid: str = "gfx_replay_keyframes_string"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._enable_gfx_replay_save = (
            self._sim.sim_config.sim_cfg.enable_gfx_replay_save
        )
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return GfxReplayMeasure.cls_uuid

    def reset_metric(self, *args, **kwargs):
        self._gfx_replay_keyframes_string = None
        self.update_metric(*args, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        if not task._is_episode_active and self._enable_gfx_replay_save:
            self._metric = (
                self._sim.gfx_replay_manager.write_saved_keyframes_to_string()
            )
        else:
            self._metric = ""

    def get_metric(self, force_get=False):
        if force_get and self._enable_gfx_replay_save:
            return (
                self._sim.gfx_replay_manager.write_saved_keyframes_to_string()
            )
        return super().get_metric()


@registry.register_measure
class ObjAtGoal(Measure):
    """
    Returns if the target object is at the goal (binary) for each of the target
    objects in the scene.
    """

    cls_uuid: str = "obj_at_goal"

    def __init__(self, *args, sim, config, task, **kwargs):
        self._config = config
        self._succ_thresh = self._config.succ_thresh
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ObjAtGoal.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                ObjectToGoalDistance.cls_uuid,
            ],
        )
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        obj_to_goal_dists = task.measurements.measures[
            ObjectToGoalDistance.cls_uuid
        ].get_metric()

        self._metric = {
            str(idx): dist < self._succ_thresh
            for idx, dist in obj_to_goal_dists.items()
        }


@registry.register_measure
class EndEffectorToGoalDistance(UsesArticulatedAgentInterface, Measure):
    cls_uuid: str = "ee_to_goal_distance"

    def __init__(self, sim, *args, **kwargs):
        self._sim = sim
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return EndEffectorToGoalDistance.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, observations, **kwargs):
        ee_pos = (
            self._sim.get_agent_data(self.agent_id)
            .articulated_agent.ee_transform()
            .translation
        )

        goals = self._sim.get_targets()[1]

        distances = np.linalg.norm(goals - ee_pos, ord=2, axis=-1)

        self._metric = {str(idx): dist for idx, dist in enumerate(distances)}


@registry.register_measure
class EndEffectorToObjectDistance(UsesArticulatedAgentInterface, Measure):
    """
    Gets the distance between the end-effector and all current target object COMs.
    """

    cls_uuid: str = "ee_to_object_distance"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        assert (
            self._config.center_cone_vector is not None
            if self._config.if_consider_gaze_angle
            else True
        ), "Want to consider grasping gaze angle but a target center_cone_vector is not provided in the config."
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return EndEffectorToObjectDistance.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, episode, **kwargs):
        ee_pos = (
            self._sim.get_agent_data(self.agent_id)
            .articulated_agent.ee_transform()
            .translation
        )

        idxs, _ = self._sim.get_targets()
        scene_pos = self._sim.get_scene_pos()
        target_pos = scene_pos[idxs]

        distances = np.linalg.norm(target_pos - ee_pos, ord=2, axis=-1)

        # Ensure the gripper maintains a desirable distance
        distances = abs(
            distances - self._config.desire_distance_between_gripper_object
        )

        if self._config.if_consider_gaze_angle:
            # Get the camera transformation
            cam_T = get_camera_transform(
                self._sim.get_agent_data(self.agent_id).articulated_agent
            )
            # Get angle between (normalized) location and the vector that the camera should
            # look at
            obj_angle = get_camera_object_angle(
                cam_T, target_pos[0], self._config.center_cone_vector
            )
            distances += obj_angle

        self._metric = {str(idx): dist for idx, dist in enumerate(distances)}


@registry.register_measure
class BaseToObjectDistance(UsesArticulatedAgentInterface, Measure):
    """
    Gets the distance between the base and all current target object COMs.
    """

    cls_uuid: str = "base_to_object_distance"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return BaseToObjectDistance.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, episode, **kwargs):
        base_pos = np.array(
            (
                self._sim.get_agent_data(
                    self.agent_id
                ).articulated_agent.base_pos
            )
        )

        idxs, _ = self._sim.get_targets()
        scene_pos = self._sim.get_scene_pos()
        target_pos = np.array(scene_pos[idxs])
        distances = np.linalg.norm(
            target_pos[:, [0, 2]] - base_pos[[0, 2]], ord=2, axis=-1
        )
        self._metric = {str(idx): dist for idx, dist in enumerate(distances)}


@registry.register_measure
class EndEffectorToRestDistance(Measure):
    """
    Distance between current end effector position and position where end effector rests within the robot body.
    """

    cls_uuid: str = "ee_to_rest_distance"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return EndEffectorToRestDistance.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, episode, task, observations, **kwargs):
        to_resting = observations[RelativeRestingPositionSensor.cls_uuid]
        rest_dist = np.linalg.norm(to_resting)

        self._metric = rest_dist


@registry.register_measure
class ReturnToRestDistance(UsesArticulatedAgentInterface, Measure):
    """
    Distance between end-effector and resting position if the articulated agent is holding the object.
    """

    cls_uuid: str = "return_to_rest_distance"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ReturnToRestDistance.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, episode, task, observations, **kwargs):
        to_resting = observations[RelativeRestingPositionSensor.cls_uuid]
        rest_dist = np.linalg.norm(to_resting)

        snapped_id = self._sim.get_agent_data(self.agent_id).grasp_mgr.snap_idx
        abs_targ_obj_idx = self._sim.scene_obj_ids[task.abs_targ_idx]
        picked_correct = snapped_id == abs_targ_obj_idx

        if picked_correct:
            self._metric = rest_dist
        else:
            T_inv = (
                self._sim.get_agent_data(self.agent_id)
                .articulated_agent.ee_transform()
                .inverted()
            )
            idxs, _ = self._sim.get_targets()
            scene_pos = self._sim.get_scene_pos()
            pos = scene_pos[idxs][0]
            pos = T_inv.transform_point(pos)

            self._metric = np.linalg.norm(task.desired_resting - pos)


@registry.register_measure
class RobotCollisions(UsesArticulatedAgentInterface, Measure):
    """
    Returns a dictionary with the counts for different types of collisions.
    """

    cls_uuid: str = "robot_collisions"

    def __init__(self, *args, sim, config, task, **kwargs):
        self._sim = sim
        self._config = config
        self._task = task
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return RobotCollisions.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self._accum_coll_info = CollisionDetails()
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        cur_coll_info = self._task.get_cur_collision_info(self.agent_id)
        self._accum_coll_info += cur_coll_info
        self._metric = {
            "total_collisions": self._accum_coll_info.total_collisions,
            "robot_obj_colls": self._accum_coll_info.robot_obj_colls,
            "robot_scene_colls": self._accum_coll_info.robot_scene_colls,
            "obj_scene_colls": self._accum_coll_info.obj_scene_colls,
        }


@registry.register_measure
class RobotForce(UsesArticulatedAgentInterface, Measure):
    """
    The amount of force in newton's accumulatively applied by the robot.
    """

    cls_uuid: str = "articulated_agent_force"

    def __init__(self, *args, sim, config, task, **kwargs):
        self._sim = sim
        self._config = config
        self._task = task
        self._count_obj_collisions = self._task._config.count_obj_collisions
        self._min_force = self._config.min_force
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return RobotForce.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self._accum_force = 0.0
        self._prev_force = None
        self._cur_force = None
        self._add_force = None
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    @property
    def add_force(self):
        return self._add_force

    def update_metric(self, *args, episode, task, observations, **kwargs):
        articulated_agent_force, _, overall_force = self._task.get_coll_forces(
            self.agent_id
        )

        if self._count_obj_collisions:
            self._cur_force = overall_force
        else:
            self._cur_force = articulated_agent_force

        if self._prev_force is not None:
            self._add_force = self._cur_force - self._prev_force
            if self._add_force > self._min_force:
                self._accum_force += self._add_force
                self._prev_force = self._cur_force
            elif self._add_force < 0.0:
                self._prev_force = self._cur_force
            else:
                self._add_force = 0.0
        else:
            self._prev_force = self._cur_force
            self._add_force = 0.0

        self._metric = {
            "accum": self._accum_force,
            "instant": self._cur_force,
        }


@registry.register_measure
class NumStepsMeasure(Measure):
    """
    The number of steps elapsed in the current episode.
    """

    cls_uuid: str = "num_steps"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return NumStepsMeasure.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self._metric = 0

    def update_metric(self, *args, episode, task, observations, **kwargs):
        self._metric += 1


@registry.register_measure
class ZeroMeasure(Measure):
    """
    The number of steps elapsed in the current episode.
    """

    cls_uuid: str = "zero"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ZeroMeasure.cls_uuid

    def reset_metric(self, *args, **kwargs):
        self._metric = 0

    def update_metric(self, *args, **kwargs):
        self._metric = 0


@registry.register_measure
class ForceTerminate(Measure):
    """
    If the accumulated force throughout this episode exceeds the limit.
    """

    cls_uuid: str = "force_terminate"

    def __init__(self, *args, sim, config, task, **kwargs):
        self._sim = sim
        self._config = config
        self._max_accum_force = self._config.max_accum_force
        self._max_instant_force = self._config.max_instant_force
        self._task = task
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ForceTerminate.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                RobotForce.cls_uuid,
            ],
        )

        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        force_info = task.measurements.measures[
            RobotForce.cls_uuid
        ].get_metric()
        accum_force = force_info["accum"]
        instant_force = force_info["instant"]
        if self._max_accum_force > 0 and accum_force > self._max_accum_force:
            rearrange_logger.debug(
                f"Force threshold={self._max_accum_force} exceeded with {accum_force}, ending episode"
            )
            self._task.should_end = True
            self._metric = True
        elif (
            self._max_instant_force > 0
            and instant_force > self._max_instant_force
        ):
            rearrange_logger.debug(
                f"Force instant threshold={self._max_instant_force} exceeded with {instant_force}, ending episode"
            )
            self._task.should_end = True
            self._metric = True
        else:
            self._metric = False


@registry.register_measure
class DidViolateHoldConstraintMeasure(UsesArticulatedAgentInterface, Measure):
    cls_uuid: str = "did_violate_hold_constraint"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return DidViolateHoldConstraintMeasure.cls_uuid

    def __init__(self, *args, sim, **kwargs):
        self._sim = sim

        super().__init__(*args, sim=sim, **kwargs)

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, **kwargs):
        self._metric = self._sim.get_agent_data(
            self.agent_id
        ).grasp_mgr.is_violating_hold_constraint()


class RearrangeReward(UsesArticulatedAgentInterface, Measure):
    """
    An abstract class defining some measures that are always a part of any
    reward function in the Habitat 2.0 tasks.
    """

    def __init__(self, *args, sim, config, task, **kwargs):
        self._sim = sim
        self._config = config
        self._task = task
        self._force_pen = self._config.force_pen
        self._max_force_pen = self._config.max_force_pen
        self._count_coll_pen = self._config.count_coll_pen
        self._max_count_colls = self._config.max_count_colls
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self._prev_count_coll = 0

        target_measure = [RobotForce.cls_uuid, ForceTerminate.cls_uuid]
        if self._want_count_coll():
            target_measure.append(RobotCollisions.cls_uuid)

        task.measurements.check_measure_dependencies(self.uuid, target_measure)

        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        reward = 0.0

        # For force collision reward (useful for dynamic simulation)
        reward += self._get_coll_reward()

        # For count-based collision reward and termination (useful for kinematic simulation)
        if self._want_count_coll():
            reward += self._get_count_coll_reward()

        # For hold constraint violation
        if self._sim.get_agent_data(
            self.agent_id
        ).grasp_mgr.is_violating_hold_constraint():
            reward -= self._config.constraint_violate_pen

        # For force termination
        force_terminate = task.measurements.measures[
            ForceTerminate.cls_uuid
        ].get_metric()
        if force_terminate:
            reward -= self._config.force_end_pen

        self._metric = reward

    def _get_coll_reward(self):
        reward = 0

        force_metric = self._task.measurements.measures[RobotForce.cls_uuid]
        # Penalize the force that was added to the accumulated force at the
        # last time step.
        reward -= max(
            0,  # This penalty is always positive
            min(
                self._force_pen * force_metric.add_force,
                self._max_force_pen,
            ),
        )
        return reward

    def _want_count_coll(self):
        """Check if we want to consider penality from count-based collisions"""
        return self._count_coll_pen != -1 or self._max_count_colls != -1

    def _get_count_coll_reward(self):
        """Count-based collision reward"""
        reward = 0

        count_coll_metric = self._task.measurements.measures[
            RobotCollisions.cls_uuid
        ]
        cur_total_colls = count_coll_metric.get_metric()["total_collisions"]

        # Check the step collision
        if (
            self._count_coll_pen != -1.0
            and cur_total_colls - self._prev_count_coll > 0
        ):
            reward -= self._count_coll_pen

        # Check the max count collision
        if (
            self._max_count_colls != -1.0
            and cur_total_colls > self._max_count_colls
        ):
            reward -= self._config.count_coll_end_pen
            self._task.should_end = True

        # update the counter
        self._prev_count_coll = cur_total_colls

        return reward


@registry.register_measure
class DoesWantTerminate(Measure):
    """
    Returns 1 if the agent has called the stop action and 0 otherwise.
    """

    cls_uuid: str = "does_want_terminate"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return DoesWantTerminate.cls_uuid

    def reset_metric(self, *args, **kwargs):
        self.update_metric(*args, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        self._metric = task.actions["rearrange_stop"].does_want_terminate


@registry.register_measure
class BadCalledTerminate(Measure):
    """
    Returns 0 if the agent has called the stop action when the success
    condition is also met or not called the stop action when the success
    condition is not met. Returns 1 otherwise.
    """

    cls_uuid: str = "bad_called_terminate"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return BadCalledTerminate.cls_uuid

    def __init__(self, config, task, *args, **kwargs):
        super().__init__(**kwargs)
        self._success_measure_name = task._config.success_measure
        self._config = config

    def reset_metric(self, *args, task, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [DoesWantTerminate.cls_uuid, self._success_measure_name],
        )
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        does_action_want_stop = task.measurements.measures[
            DoesWantTerminate.cls_uuid
        ].get_metric()
        is_succ = task.measurements.measures[
            self._success_measure_name
        ].get_metric()

        self._metric = (not is_succ) and does_action_want_stop


@registry.register_measure
class RuntimePerfStats(Measure):
    cls_uuid: str = "habitat_perf"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return RuntimePerfStats.cls_uuid

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._sim.enable_perf_logging()
        self._disable_logging = config.disable_logging
        super().__init__()

    def reset_metric(self, *args, **kwargs):
        self._metric_queue = defaultdict(deque)
        self._metric = {}

    def update_metric(self, *args, task, **kwargs):
        for k, v in self._sim.get_runtime_perf_stats().items():
            self._metric_queue[k].append(v)
        if self._disable_logging:
            self._metric = {}
        else:
            self._metric = {
                k: np.mean(v) for k, v in self._metric_queue.items()
            }


@registry.register_sensor
class HasFinishedOracleNavSensor(UsesArticulatedAgentInterface, Sensor):
    """
    Returns 1 if the agent has finished the oracle nav action. Returns 0 otherwise.
    """

    cls_uuid: str = "has_finished_oracle_nav"

    def __init__(self, sim, config, *args, task, **kwargs):
        self._task = task
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return HasFinishedOracleNavSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(shape=(1,), low=0, high=1, dtype=np.float32)

    def get_observation(self, observations, episode, *args, **kwargs):
        if self.agent_id is not None:
            use_k = f"agent_{self.agent_id}_oracle_nav_action"
        else:
            use_k = "oracle_nav_action"

        if use_k not in self._task.actions:
            return np.array(False, dtype=np.float32)[..., None]
        else:
            nav_action = self._task.actions[use_k]
            return np.array(nav_action.skill_done, dtype=np.float32)[..., None]


@registry.register_sensor
class HasFinishedHumanoidPickSensor(UsesArticulatedAgentInterface, Sensor):
    """
    Returns 1 if the agent has finished the oracle nav action. Returns 0 otherwise.
    """

    cls_uuid: str = "has_finished_human_pick"

    def __init__(self, sim, config, *args, task, **kwargs):
        self._task = task
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return HasFinishedHumanoidPickSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(shape=(1,), low=0, high=1, dtype=np.float32)

    def get_observation(self, observations, episode, *args, **kwargs):
        if self.agent_id is not None:
            use_k = f"agent_{self.agent_id}_humanoid_pick_action"
        else:
            use_k = "humanoid_pick_action"

        nav_action = self._task.actions[use_k]

        return np.array(nav_action.skill_done, dtype=np.float32)[..., None]


@registry.register_sensor
class ArmDepthBBoxSensor(UsesArticulatedAgentInterface, Sensor):
    """Bounding box sensor to check if the object is in frame"""

    cls_uuid: str = "arm_depth_bbox_sensor"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim
        self._height = config.height
        self._width = config.width

    def _get_uuid(self, *args, **kwargs):
        return ArmDepthBBoxSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(
                config.height,
                config.width,
                1,
            ),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def _get_bbox(self, img):
        """Simple function to get the bounding box, assuming that only one object of interest in the image"""
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax

    def get_observation(self, observations, episode, task, *args, **kwargs):
        # Get a correct observation space
        if self.agent_id is None:
            target_key = "articulated_agent_arm_panoptic"
            assert target_key in observations
        else:
            target_key = (
                f"agent_{self.agent_id}_articulated_agent_arm_panoptic"
            )
            assert target_key in observations

        img_seg = observations[target_key]

        # Check the size of the observation
        assert (
            img_seg.shape[0] == self._height
            and img_seg.shape[1] == self._width
        )

        # Check if task has the attribute of the abs_targ_idx
        assert hasattr(task, "abs_targ_idx")

        # Get the target from sim, and ensure that the index is offset
        tgt_idx = (
            self._sim.scene_obj_ids[task.abs_targ_idx]
            + self._sim.habitat_config.object_ids_start
        )
        tgt_mask = (img_seg == tgt_idx).astype(int)

        # Get the bounding box
        bbox = np.zeros(tgt_mask.shape)
        if np.sum(tgt_mask) != 0:
            rmin, rmax, cmin, cmax = self._get_bbox(tgt_mask)
            bbox[rmin:rmax, cmin:cmax] = 1.0

        return np.float32(bbox)
