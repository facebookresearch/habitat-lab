#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
from collections import defaultdict, deque

import magnum as mn
import numpy as np
import quaternion
import torch
from gym import spaces

from habitat.articulated_agents.humanoids import KinematicHumanoid
from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.datasets.rearrange.samplers.receptacle import find_receptacles
from habitat.tasks.nav.nav import PointGoalSensor
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.utils import (
    CollisionDetails,
    UsesArticulatedAgentInterface,
    batch_transform_point,
    get_angle_to_pos,
    get_angle_to_pos_xyz,
    get_camera_object_angle,
    get_camera_transform,
    rearrange_logger,
)
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    angle_between_quaternions,
    quat_to_euler,
)
from habitat.utils.rotation_utils import (
    convert_conventions,
    extract_roll_pitch_yaw,
    transform_position,
)


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
        # TODO: This is a hack. For some reason _polar_pos in overriden by the other
        # agent.
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

    def _get_observation_space(self, *args, **kwargs):
        if self.config.only_one_target or self.config.use_noise_target:
            n_targets = 1
        else:
            n_targets = self._task.get_n_targets()
        return spaces.Box(
            shape=(3 * n_targets,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def xyz_T_hab(self, tf, reverse=False):
        """
        Convert from habitat -> real-world xyz coordinates
        If reverse = True, converts from real-world xyz -> habitat coordinates
        """
        xyz_T_hab_rot = mn.Matrix4(
            [
                [0.0, 0.0, -1.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ).transposed()
        # xyz_T_hab_rot = mn.Matrix4(
        #     [
        #         [1.0, 0.0, 0.0, 0.0],
        #         [0.0, 0.0, 1.0, 0.0],
        #         [0.0, -1.0, 0.0, 0.0],
        #         [0.0, 0.0, 0.0, 1.0],
        #     ]
        # ).transposed()
        # xyz_T_hab_rot = mn.Matrix4(  # most likely wrong
        #     [
        #         [-1.0, 0.0, 0.0, 0.0],
        #         [0.0, 0.0, 1.0, 0.0],
        #         [0.0, 1.0, 0.0, 0.0],
        #         [0.0, 0.0, 0.0, 1.0],
        #     ]
        # ).transposed()
        tf_untranslate = mn.Matrix4.translation(-tf.translation)
        if not reverse:
            rot = xyz_T_hab_rot
        else:
            rot = xyz_T_hab_rot.inverted()
        tf_retranslate = mn.Matrix4.translation(
            rot.transform_point(tf.translation)
        )

        tf_new = tf_untranslate @ tf
        tf_new = xyz_T_hab_rot.inverted() @ tf_new
        tf_new = tf_retranslate @ tf_new
        return tf_new

    def get_observation_real(self, task):
        _, sim_global_T_obj_pos = self._sim.get_targets()
        global_T_obj_pos_YXZ = sim_global_T_obj_pos[task.targ_idx]
        global_T_obj_hab = mn.Vector3(*global_T_obj_pos_YXZ)
        global_T_obj_std = convert_conventions(global_T_obj_hab)

        global_T_ee = self._sim.articulated_agent.ee_transform()
        ee_T_obj_XYZ = global_T_ee.inverted().transform_point(global_T_obj_std)
        return np.array(ee_T_obj_XYZ, dtype=np.float32)

    def get_observation(self, observations, episode, task, *args, **kwargs):
        if self.config.use_real_world_conventions:
            return self.get_observation_real(task)

        global_T_ee = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.ee_transform()

        if self.config.use_ee_T_target:
            # xyz_global_T_ee = self.xyz_T_hab(global_T_ee)
            # xyz_ee_T_global = xyz_global_T_ee.inverted()

            # _, global_T_target_pos = self._sim.get_targets()
            # global_T_target = mn.Matrix4.translation(
            #     mn.Vector3(global_T_target_pos[task.targ_idx])
            # )
            # xyz_global_T_target = self.xyz_T_hab(global_T_target)
            # xyz_global_T_target_pos = xyz_global_T_target.translation

            # xyz_ee_T_target_pos = xyz_ee_T_global.transform_point(
            #     xyz_global_T_target_pos
            # )

            # return np.array(xyz_ee_T_target_pos, dtype=np.float32)

            xyz_T_hab_rot = mn.Matrix4(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ).transposed()

            global_T = global_T_ee
            # print("global_T_ee: ", global_T_ee)

            if self.config.use_base_transform:
                base_T = self._sim.get_agent_data(
                    self.agent_id
                ).articulated_agent.base_transformation
                global_T_base = mn.Matrix4(base_T)
                # print("global_T_base: ", base_T)

                # Make the ee location as the base location
                base_T.translation = global_T.translation
                global_T = mn.Matrix4(base_T)

                base_T_xyz = xyz_T_hab_rot @ global_T

            # Inversion
            T_inv = global_T.inverted()
            T_inv_xyz = base_T_xyz.inverted()

            # Get the target position
            _, pos = self._sim.get_targets()
            base_T_target_pos = mn.Vector3([1.0, 0.0, 0.5])

            global_T_base_xyz = xyz_T_hab_rot @ global_T_base
            pos_xyz = xyz_T_hab_rot.transform_point(
                mn.Vector3(pos[task.targ_idx])
            )
            pos_xyz = np.array([[pos_xyz.x, pos_xyz.y, pos_xyz.z]])
            # print("pos_xyz: ", pos_xyz)
            # print("global_T_base: ", global_T_base)
            # print("global_T_base_xyz: ", global_T_base_xyz)
            # print("base_T_target_pos: ", base_T_target_pos)

            global_T_target_base = mn.Matrix4()
            global_T_target_base.translation = global_T_base.translation

            global_T_target = global_T_base_xyz.transform_point(
                base_T_target_pos
            )
            # print("global_T_target: ", global_T_target)
            # print("pos: ", pos)
            self._sim.viz_ids["place_tar"] = self._sim.visualize_position(
                global_T_target,
                self._sim.viz_ids["place_tar"],
            )

            # print("global_T_target: ", pos)
            bullet_ee_xyz, _ = task.actions["arm_action"].get_ee_pose()
            print("ee_pose: ", bullet_ee_xyz)

            # [x,y,z]
            # x: ee as origin, front is +; back is -
            # y: ee as origin, left is +; right is -
            # z: ee as origin, up is +; down is -
            pos_array = batch_transform_point(pos, T_inv, np.float32)[
                [task.targ_idx]
            ].reshape(-1)
            pos_array_xyz = batch_transform_point(
                pos_xyz, T_inv_xyz, np.float32
            )[[task.targ_idx]].reshape(-1)
            print("obj_goal_sensor: ", pos_array)
            print("obj_goal_sensor_xyz: ", pos_array_xyz)
            return np.array(pos_array, dtype=np.float32)
        else:
            global_T = global_T_ee
            # print("global_T_ee: ", global_T_ee)

            if self.config.use_base_transform:
                base_T = self._sim.get_agent_data(
                    self.agent_id
                ).articulated_agent.base_transformation
                # print("global_T_base: ", base_T)

                # Make the ee location as the base location
                base_T.translation = global_T.translation
                global_T = mn.Matrix4(base_T)

            # Inversion
            T_inv = global_T.inverted()

            # Get the target position
            _, pos = self._sim.get_targets()
            # print("global_T_target: ", pos)
            # bullet_ee_xyz, _ = task.actions["arm_action"].get_ee_pose()
            # print("ee_pose: ", bullet_ee_xyz)

            # [x,y,z]
            # x: ee as origin, front is +; back is -
            # y: ee as origin, left is +; right is -
            # z: ee as origin, up is +; down is -
            if self.config.use_noise_target:
                pos_array = batch_transform_point(
                    np.array([task.noise_target_location]), T_inv, np.float32
                )[0].reshape(-1)
            else:
                if self.config.only_one_target:
                    pos_array = batch_transform_point(pos, T_inv, np.float32)[
                        [task.targ_idx]
                    ].reshape(-1)
                else:
                    pos_array = batch_transform_point(
                        pos, T_inv, np.float32
                    ).reshape(-1)
            # print("obj_goal_sensor: ", pos_array)
            return np.array(pos_array, dtype=np.float32)


@registry.register_sensor
class DistanceGoalSensor(UsesArticulatedAgentInterface, MultiObjSensor):
    """
    Relative to the end effector, but with norm
    """

    cls_uuid: str = "distance_goal_sensor"

    def _get_observation_space(self, *args, **kwargs):
        if self.config.only_one_target:
            n_targets = 1
        else:
            n_targets = self._task.get_n_targets()
        return spaces.Box(
            shape=(n_targets,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, task, *args, **kwargs):
        global_T = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.ee_transform()
        T_inv = global_T.inverted()

        _, pos = self._sim.get_targets()
        if self.config.only_one_target:
            xyz = batch_transform_point(pos, T_inv, np.float32)[
                [task.targ_idx]
            ].reshape(-1)
            return np.linalg.norm(xyz, keepdims=True)
        else:
            xyz = batch_transform_point(pos, T_inv, np.float32).reshape(-1)
            xyz_dis = []
            for i in range(self._task.get_n_targets()):
                xyz_dis.append(np.linalg.norm(xyz[i * 3 : (i + 1) * 3]))
            return np.array(xyz_dis, dtype=np.float32)


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

    def get_observation_real(self):
        joints_pos = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.get_arm_joint_positions()
        return joints_pos

    def get_observation(self, observations, episode, *args, **kwargs):
        if self.config.use_real_world_conventions:
            joints_pos = self.get_observation_real()
        else:
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
class EEPoseSensor(UsesArticulatedAgentInterface, Sensor):
    cls_uuid: str = "ee_pose"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return EEPoseSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(6,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation_real(self):
        global_T_ee = self._sim.articulated_agent.ee_transform()
        local_ee_pos = global_T_ee.translation
        local_ee_rpy = extract_roll_pitch_yaw(global_T_ee.rotation())

        return np.array([*local_ee_pos, *local_ee_rpy], dtype=np.float32)

    def get_observation(self, observations, episode, task, *args, **kwargs):
        if self.config.use_real_world_conventions:
            return self.get_observation_real()

        bullet_ee_xyz, bullet_ee_rpy = task.actions["arm_action"].get_ee_pose()

        return np.array([*bullet_ee_xyz, *bullet_ee_rpy], dtype=np.float32)


@registry.register_sensor
class ReceptacleBBoxSensor(UsesArticulatedAgentInterface, Sensor):
    cls_uuid: str = "receptacle_bbox"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ReceptacleBBoxSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(6,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, task, *argsr, **kwargs):
        receptacles = find_receptacles(self._sim)
        for receptacle in receptacles:
            print(
                "receptacle q4eqasd: ",
                receptacle.name,
                receptacle.bounds,
            )
            self._sim.viz_ids["bounds_min"] = self._sim.visualize_position(
                receptacle.bounds.bounds.min,
            )
            self._sim.viz_ids["bounds_max"] = self._sim.visualize_position(
                receptacle.bounds.bounds.max,
            )
            print("vis ids: ", self._sim.viz_ids)
        return np.zeros(6)


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
class RelativeInitialEEOrientationSensor(
    UsesArticulatedAgentInterface, Sensor
):
    cls_uuid: str = "relative_initial_ee_orientation"

    def _get_uuid(self, *args, **kwargs):
        return RelativeInitialEEOrientationSensor.cls_uuid

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim
        self._use_smallest_angle = config.use_smallest_angle

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(1,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, task, *args, **kwargs):
        _, ee_orientation = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.get_ee_local_pose()
        return np.array(
            [
                angle_between_quaternions(
                    task.init_ee_orientation, ee_orientation
                )
            ],
            dtype=np.float32,
        )


@registry.register_sensor
class RelativeTargetObjectOrientationSensor(
    UsesArticulatedAgentInterface, Sensor
):
    cls_uuid: str = "relative_target_object_orientation"

    def _get_uuid(self, *args, **kwargs):
        return RelativeTargetObjectOrientationSensor.cls_uuid

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(1,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, task, *args, **kwargs):
        agent = self._sim.get_agent_data(self.agent_id).articulated_agent

        _, ee_orientation = agent.get_ee_local_pose()

        # The target object orientation is initial object orientation
        target_object_orientation = task.target_obj_orientation

        offset = 0.0
        if self.config.offset_yaw:
            # If we want to remove the angle in yaw direction
            # Get all the transformations
            base_transform = agent.base_transformation
            ee_transform = agent.ee_transform()
            armbase_transform = agent.sim_obj.get_link_scene_node(
                0
            ).transformation
            # Offset the base based on the armbase transform
            base_transform.translation = base_transform.transform_point(
                (base_transform.inverted() @ armbase_transform).translation
            )
            # Do transformation
            ee_position = (
                base_transform.inverted() @ ee_transform
            ).translation
            # Process the ee_position input
            ee_position = abs(np.array(ee_position))
            # If norm is too small, then we do not do anything
            if np.linalg.norm(ee_position[[0, 1]]) > 0.1:
                offset = abs(get_angle_to_pos_xyz(ee_position))

        return np.array(
            [
                angle_between_quaternions(
                    target_object_orientation, ee_orientation
                )
                - offset
            ],
            dtype=np.float32,
        )


@registry.register_sensor
class TopDownOrSideGraspingSensor(UsesArticulatedAgentInterface, Sensor):
    cls_uuid: str = "topdown_or_side_grasping"

    def _get_uuid(self, *args, **kwargs):
        return TopDownOrSideGraspingSensor.cls_uuid

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(1,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, task, *args, **kwargs):
        # Get the current agent
        agent = self._sim.get_agent_data(self.agent_id).articulated_agent
        ee_T = agent.ee_transform()
        base_T = agent.base_transformation
        base_to_ee_T = base_T.inverted() @ ee_T
        target_vector = np.array([0, 0, 1.0])
        dir_vector = np.array(base_to_ee_T.transform_vector(target_vector))
        # Get the target vector
        if task.grasping_type == "topdown":
            delta = 1.0 - abs(dir_vector[2])
        elif task.grasping_type == "side":
            delta = abs(dir_vector[2])
        else:
            raise ValueError(f"Unknown grasping type {task.grasping_type}")
        return np.array(
            [delta],
            dtype=np.float32,
        )


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
        if self._sim.use_real_world_conventions:
            goal_pos = [
                np.array(convert_conventions(mn.Vector3(*goal)))
                for goal in goal_pos
            ]
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
        if self._sim.use_real_world_conventions:
            goals = np.array(
                [convert_conventions(mn.Vector3(*goal)) for goal in goals]
            )

        distances = np.linalg.norm(goals - np.array(ee_pos), ord=2, axis=-1)

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

        if self._config.get("if_consider_detected_portion", False):
            bbox = kwargs["observations"]["arm_depth_bbox_sensor"]
            # Compute the detected portion in the bounding box sensor
            # Since this is the distance, the smaller the value is, the better
            distances += 1.0 - np.mean(bbox)

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

        # For force collision reward (userful for dynamic simulation)
        reward += self._get_coll_reward()

        # For count-based collision reward and termination (userful for kinematic simulation)
        if self._want_count_coll():
            reward += self._get_count_coll_reward(observations)

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

    def _get_count_coll_reward(self, observations):
        """Count-based collision reward"""
        reward = 0

        count_coll_metric = self._task.measurements.measures[
            RobotCollisions.cls_uuid
        ]
        cur_total_colls = count_coll_metric.get_metric()["total_collisions"]

        contact_test_collisions = int(observations.get("collided", False))

        # Check the step collision
        if (
            self._count_coll_pen != -1.0
            and cur_total_colls - self._prev_count_coll > 0
        ):
            reward -= self._count_coll_pen

        # add penalty for contact test collisions
        if self._count_coll_pen != -1.0 and contact_test_collisions > 0:
            reward -= self._count_coll_pen

        # Check the max count collision
        if (
            self._max_count_colls != -1.0
            and cur_total_colls > self._max_count_colls
        ):
            reward -= self._config.count_coll_end_pen
            rearrange_logger.debug(f"Exceeded max collisions, ending episode")
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
        self._noise = config.get("noise", 0)

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

        # Don't show bounding box with some probability
        if np.random.rand() < self._noise:
            return np.float32(bbox)

        if np.sum(tgt_mask) != 0:
            rmin, rmax, cmin, cmax = self._get_bbox(tgt_mask)
            bbox[rmin:rmax, cmin:cmax] = 1.0

        return np.float32(bbox)


@registry.register_sensor
class ArmRGBPretrainVisualFeatureSensor(UsesArticulatedAgentInterface, Sensor):
    """Pretrained visual feature sensor for arm rgb camera"""

    cls_uuid: str = "arm_rgb_pretrain_visual_feature_sensor"

    def __init__(self, sim, config, *args, **kwargs):
        try:
            from transformers import (
                AutoModel,
                AutoProcessor,
                SiglipVisionConfig,
            )
        except ImportError as e:
            raise RuntimeError(
                "Failed to import transformer package. Please install HuggingFace transformers by the following: pip install git+https://github.com/huggingface/transformers"
            ) from e
        self._device = (
            torch.device("cuda")
            if torch.cuda.is_available() and not config.force_to_use_cpu
            else torch.device("cpu")
        )
        configuration = SiglipVisionConfig()
        self._feature_dim = configuration.hidden_size  # 768
        self._model_name = "google/siglip-base-patch16-224"
        super().__init__(config=config)
        self._sim = sim
        # Load the model based on HuggingFace's transformers
        # Use bfloat16 to speed up inference
        self._model = AutoModel.from_pretrained(
            self._model_name, torch_dtype=torch.bfloat16
        ).to(self._device)
        self._processor = AutoProcessor.from_pretrained(self._model_name)

    def _get_uuid(self, *args, **kwargs):
        return ArmRGBPretrainVisualFeatureSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(self._feature_dim,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, task, *args, **kwargs):
        # Get a correct observation space
        if self.agent_id is None:
            target_key = "articulated_agent_arm_rgb"
            assert target_key in observations
        else:
            target_key = f"agent_{self.agent_id}_articulated_agent_arm_rgb"
            assert target_key in observations

        # Get image
        rgb_image = observations[target_key]

        # Feed the image into the model
        rgb_image_input = self._processor(
            images=rgb_image, return_tensors="pt"
        )
        rgb_image_input = rgb_image_input.to(self._device)
        rgb_image_input["pixel_values"] = rgb_image_input[
            "pixel_values"
        ].bfloat16()
        with torch.inference_mode():
            image_features = self._model.get_image_features(**rgb_image_input)

        # Move the features back to cpu
        image_features = image_features.to("cpu")
        image_features = image_features.float()
        image_features = np.array(image_features.squeeze())

        return image_features


@registry.register_sensor
class PretrainTextualFeatureGoalSensor(UsesArticulatedAgentInterface, Sensor):
    """Pretrained visual feature sensor for arm rgb camera"""

    cls_uuid: str = "pretrain_textual_feature_goal_sensor"

    def __init__(self, sim, config, *args, **kwargs):
        try:
            from transformers import (
                AutoModel,
                AutoTokenizer,
                SiglipVisionConfig,
            )
        except ImportError as e:
            raise RuntimeError(
                "Failed to import transformer package. Please install HuggingFace transformers by the following: pip install git+https://github.com/huggingface/transformers"
            ) from e
        self._device = (
            torch.device("cuda")
            if torch.cuda.is_available() and not config.force_to_use_cpu
            else torch.device("cpu")
        )
        configuration = SiglipVisionConfig()
        self._feature_dim = configuration.hidden_size  # 768
        self._model_name = "google/siglip-base-patch16-224"
        super().__init__(config=config)
        self._sim = sim
        # Load the model based on HuggingFace's transformers
        # Use bfloat16 to speed up inference
        self._model = AutoModel.from_pretrained(
            self._model_name, torch_dtype=torch.bfloat16
        ).to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(
            "google/siglip-base-patch16-224"
        )
        self._goal_name = None
        self._goal_pos = None
        self._goal_textual_feature = None
        self._diff_treshold = 0.01
        self._target_receptacle_names = [
            "refrigerator",
            "cupboard",
            "counter",
            "table",
            "tvstand",
            "sofa",
            "cabinet",
            "chair",
        ]
        self._pre_fix = "place an object on the "

        self._goal_obj_name = None
        self._goal_obj_textual_feature = None
        self._target_object_names = [
            "mug",
            "bowl",
            "banana",
            "tomato_soup_can",
            "master_chef_can",
            "sugar_box",
            "tuna_fish_can",
            "bleach_cleanser",
            "preach",
            "lemon",
        ]
        self._use_features = self.config.use_features

    def _get_uuid(self, *args, **kwargs):
        return PretrainTextualFeatureGoalSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        if len(self.config.use_features) == 2:
            return spaces.Box(
                shape=(self._feature_dim * 2,),
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                dtype=np.float32,
            )
        else:
            return spaces.Box(
                shape=(self._feature_dim,),
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                dtype=np.float32,
            )

    def _get_goal_text(self, targ_idx):
        # Cache the name of the receptacle
        _, goal_pos = self._sim.get_targets()
        if (
            self._goal_pos is not None
            and np.linalg.norm(
                goal_pos[targ_idx, [0, 2]] - self._goal_pos[targ_idx, [0, 2]]
            )
            < self._diff_treshold
        ):
            return self._goal_name

        min_dis = float("inf")
        goal_name = "None"
        for rep_name in self._sim.receptacles:
            distance = np.linalg.norm(
                np.array(self._sim.receptacles[rep_name].center())[[0, 2]]
                - goal_pos[targ_idx, [0, 2]]
            )
            if distance < min_dis:
                min_dis = distance  # type: ignore
                goal_name = rep_name

        self._goal_pos = goal_pos

        return goal_name

    def _filter_receptacle_name(self, raw_name):
        candidate_name = [
            name
            for name in self._target_receptacle_names
            if name in raw_name.lower().replace("_", "")
        ]

        # Handle special case
        if len(candidate_name) == 0:
            return self._pre_fix + "furniture"
        # Only return the first one
        candidate_name = candidate_name[0]
        # Handle special case
        if candidate_name == "cabinet":
            return self._pre_fix + "chair"
        else:
            return self._pre_fix + candidate_name

    def _get_object_name(self, targ_idx):
        """Get the object name."""
        # Only support the first one
        object_name = [key for key, value in self._sim._targets.items()][
            targ_idx
        ]
        object_name = object_name.split("_")
        object_name = object_name[1:-1]
        object_name = "_".join(object_name)
        return object_name

    def _get_text_feature(self, text):
        # Prepare the model input
        inputs = self._tokenizer(
            [self._filter_receptacle_name(text)],
            padding="max_length",
            return_tensors="pt",
        )
        inputs = inputs.to(self._device)
        # Get text feature
        with torch.inference_mode():
            text_features = self._model.get_text_features(**inputs)

        # Move the features back to cpu
        text_features = text_features.to("cpu")
        text_features = text_features.float()
        text_features = np.array(text_features.squeeze())
        return text_features

    def get_observation(self, observations, episode, task, *args, **kwargs):
        # Get a target receptacle name
        rep_name = self._get_goal_text(task.targ_idx)

        # Get a target object name
        obj_name = self._get_object_name(task.targ_idx)
        if (
            self._goal_name == rep_name
            and "rep" in self._use_features
            and len(self._use_features) == 1
        ):
            return self._goal_textual_feature

        if (
            self._goal_obj_name == obj_name
            and "obj" in self._use_features
            and len(self._use_features) == 1
        ):
            return self._goal_obj_textual_feature

        if (
            self._goal_name == rep_name
            and self._goal_obj_name == obj_name
            and len(self._use_features) == 2
        ):
            return np.concatenate(
                [self._goal_textual_feature, self._goal_obj_textual_feature]
            )

        return_feature = []
        # Get the receptacle text feature
        if "rep" in self._use_features:
            text_features = self._get_text_feature(rep_name)
            self._goal_textual_feature = text_features
            self._goal_name = rep_name
            return_feature.append(text_features)

        if "obj" in self._use_features:
            # Get the object text feature
            obj_text_features = self._get_text_feature(obj_name)
            self._goal_obj_textual_feature = obj_text_features
            self._goal_obj_name = obj_name
            return_feature.append(obj_text_features)

        return np.concatenate(return_feature, axis=0)


@registry.register_sensor
class ArmReceptacleSemanticSensor(UsesArticulatedAgentInterface, Sensor):
    """Semantic sensor for the target place receptacle"""

    cls_uuid: str = "arm_receptacle_semantic_sensor"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim
        self._height = config.height
        self._width = config.width

        self._rep_name = None
        self._rep_pos = None
        self._diff_treshold = 0.01
        self._rep_name_to_semantic_id = {
            "kitchen_counter": 33,
            "refrigerator": 1,
            "refrigerator": 1,
        }

    def _get_uuid(self, *args, **kwargs):
        return ArmReceptacleSemanticSensor.cls_uuid

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

    def _get_rep_text(self, targ_idx):
        # Cache the name of the receptacle
        _, rep_pos = self._sim.get_targets()
        if (
            self._rep_pos is not None
            and np.linalg.norm(
                rep_pos[targ_idx, [0, 2]] - self._rep_pos[targ_idx, [0, 2]]
            )
            < self._diff_treshold
        ):
            return self._rep_name

        min_dis = float("inf")
        find_rep_name = "None"
        for rep_name in self._sim.receptacles_id:
            distance = np.linalg.norm(
                np.array(self._sim.receptacles_id[rep_name][1].center())[
                    [0, 2]
                ]
                - rep_pos[targ_idx, [0, 2]]
            )
            if distance < min_dis:
                min_dis = distance  # type: ignore
                find_rep_name = rep_name
        self._rep_pos = rep_pos
        return find_rep_name

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

        # Get the target receptacle name
        rep_name = self._get_rep_text(task.targ_idx)
        self._rep_name = rep_name

        # Get the target mask
        tgt_id = self._sim.receptacles_id[rep_name][0]
        tgt_mask = (img_seg == tgt_id).astype(int)

        return np.float32(tgt_mask)


@registry.register_sensor
class JawReceptacleSemanticSensor(ArmReceptacleSemanticSensor):
    """Semantic sensor for the target place receptacle"""

    cls_uuid: str = "jaw_receptacle_semantic_sensor"

    def _get_uuid(self, *args, **kwargs):
        return JawReceptacleSemanticSensor.cls_uuid

    def get_observation(self, observations, episode, task, *args, **kwargs):
        # Get a correct observation space
        if self.agent_id is None:
            target_key = "articulated_agent_jaw_panoptic"
            assert target_key in observations
        else:
            target_key = (
                f"agent_{self.agent_id}_articulated_agent_jaw_panoptic"
            )
            assert target_key in observations

        img_seg = observations[target_key]

        # Check the size of the observation
        assert (
            img_seg.shape[0] == self._height
            and img_seg.shape[1] == self._width
        )

        # Get the target receptacle name
        rep_name = self._get_rep_text(task.targ_idx)
        self._rep_name = rep_name

        # Get the target mask
        tgt_id = self._sim.receptacles_id[rep_name][0]
        tgt_mask = (img_seg == tgt_id).astype(int)

        return np.float32(tgt_mask)
