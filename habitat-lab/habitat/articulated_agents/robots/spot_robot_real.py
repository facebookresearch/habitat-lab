# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
from typing import Dict, List, Optional, Set, Tuple

import magnum as mn
import numpy as np
import quaternion

from habitat.articulated_agents.mobile_manipulator import (
    ArticulatedAgentCameraParams,
    MobileManipulator,
    MobileManipulatorParams,
)
from habitat.articulated_agents.robots.spot_robot import SpotRobot
from habitat.utils.rotation_utils import *


class SpotRobotReal(SpotRobot):
    def __init__(
        self, agent_cfg, sim, limit_robo_joints=True, fixed_base=False
    ):
        super().__init__(
            agent_cfg,
            sim,
            limit_robo_joints,
            fixed_base,
        )
        # self.roll_offset = mn.Matrix4.rotation(
        #     mn.Rad(np.pi / 2), mn.Vector3(1, 0, 0)
        # ).rotation()

        self.roll_offset = mn.Matrix3x3(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0],
                ]
            )
        )

    @property
    def base_transformation_YZX(self):
        add_rot = mn.Matrix4.rotation(
            mn.Rad(-np.pi / 2), mn.Vector3(1.0, 0, 0)
        )
        return self.sim_obj.transformation @ add_rot

    @property
    def base_transformation(self):
        global_T_base_raw_std = convert_conventions(
            self.sim_obj.transformation
        )
        global_T_base_std = mn.Matrix4().from_(
            rotation_scaling=global_T_base_raw_std.rotation()
            @ self.roll_offset.transposed(),
            translation=global_T_base_raw_std.translation,
        )
        return global_T_base_std

    def ee_transform_YZX(self, ee_index: int = 0) -> mn.Matrix4:
        if ee_index >= len(self.params.ee_links):
            raise ValueError(
                "The current manipulator does not have enough end effectors"
            )
        ef_link_transform = self.sim_obj.get_link_scene_node(
            self.params.ee_links[ee_index]
        ).transformation
        ef_link_transform.translation = ef_link_transform.transform_point(
            self.ee_local_offset(ee_index)
        )
        return ef_link_transform

    def ee_transform(self, ee_index: int = 0) -> mn.Matrix4:
        # global_T_ee_raw_hab = self.ee_transform_YZX(ee_index)
        global_T_ee_raw_hab = self.sim_obj.get_link_scene_node(
            ee_index
        ).transformation
        global_T_ee_raw_hab.translation = global_T_ee_raw_hab.transform_point(
            mn.Vector3(0.08, 0, 0)
        )
        global_T_ee_std = convert_conventions(global_T_ee_raw_hab)
        return global_T_ee_std

    def get_ee_local_pose(
        self, ee_index: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the local pose of the end-effector.

        :param ee_index: the end effector index for which we want the link transform
        """
        if ee_index >= len(self.params.ee_links):
            raise ValueError(
                "The current manipulator does not have enough end effectors"
            )

        ee_transform = self.ee_transform()
        base_transform = self.base_transformation
        base_T_ee_transform = base_transform.inverted() @ ee_transform

        # Get the local ee location (x,y,z)
        local_ee_location = base_T_ee_transform.translation

        # Get the local ee orientation (roll, pitch, yaw)
        local_ee_quat = quaternion.from_rotation_matrix(
            base_T_ee_transform.rotation()
        )

        return np.array(local_ee_location), local_ee_quat

    def get_xy_yaw(self):
        global_T_base_std = self.base_transformation

        base_position = global_T_base_std.translation
        base_rotation_rpy = extract_roll_pitch_yaw(
            global_T_base_std.rotation()
        )
        return np.array(
            [base_position[0], base_position[1], base_rotation_rpy[-1]]
        )

    def set_robot_base_transform(self, global_T_base_std):
        global_T_base_raw_std = mn.Matrix4().from_(
            rotation_scaling=global_T_base_std.rotation() @ self.roll_offset,
            translation=global_T_base_std.translation,
        )
        global_T_base_raw_std.translation += mn.Vector3(0, 0, 0.48)
        self.sim_obj.transformation = convert_conventions(
            global_T_base_raw_std, reverse=True
        )

    def set_base_position(self, real_x_pos, real_y_pos, yaw, relative=False):
        global_T_base_std = self.base_transformation

        if relative:
            relative_trans = mn.Vector3(real_x_pos, real_y_pos, 0.0)
            new_translation = global_T_base_std.transform_point(relative_trans)

            yaw_offset = mn.Matrix4.rotation(
                mn.Rad(yaw), mn.Vector3(0, 0, 1)
            ).rotation()
            new_global_T_base_std = mn.Matrix4().from_(
                translation=new_translation,
                rotation_scaling=yaw_offset @ global_T_base_std.rotation(),
            )
        else:
            new_global_T_base_std = mn.Matrix4.rotation(
                mn.Rad(yaw), mn.Vector3(0, 0, 1)
            )
            new_global_T_base_std.translation = mn.Vector3(
                real_x_pos, real_y_pos, global_T_base_std.translation.z
            )
        self.set_robot_base_transform(new_global_T_base_std)

    def set_base_velocity(self, x_vel, y_vel, ang_vel, vel_time):
        x_pos = x_vel * vel_time
        y_pos = y_vel * vel_time
        ang = ang_vel * vel_time
        return self.set_base_position(x_pos, y_pos, ang)

    def get_arm_joint_positions(
        self,
    ):
        return self.arm_joint_pos

    def set_arm_joint_positions(self, positions):
        """
        Joint angles in radians: sh0, sh1, hr1, el0, el1, wr0, wr1
        """
        self.arm_joint_pos = positions
