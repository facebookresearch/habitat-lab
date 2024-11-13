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
    def transform_to_real(self, sim_tform_matrix):
        real_translation = transform_position(
            sim_tform_matrix.translation, direction="sim_to_real"
        )
        sim_rot_euler = matrix_to_euler(sim_tform_matrix.rotation())
        real_rotation = transform_rotation(
            sim_rot_euler, rotation_format="matrix"
        )
        real_tform_matrix = create_tform_matrix(
            real_translation, real_rotation, rotation_format="mn"
        )
        return real_tform_matrix

    def global_T_body(self):
        return self.transform_to_real(self.sim_obj.transformation)

    def get_body_position(self):
        """
        Returns translation in real-world conventions [x,y,z] where +x is forward, +y is left, and +z is up
        """
        global_T_body = self.global_T_body()
        return global_T_body.translation

    def get_body_rotation(self, rotation_format="euler"):
        """
        Returns translation in real-world conventions [r,p,y] where +r is +y is +z is
        """
        global_T_body = self.global_T_body()
        real_rotation_matrix = np.array(global_T_body.rotation())
        if rotation_format.lower() == "euler":
            new_rotation = matrix_to_euler(real_rotation_matrix)
        elif rotation_format.lower() == "matrix":
            new_rotation = real_rotation_matrix
        elif rotation_format.lower() == "quaternion":
            new_rotation = matrix_to_quaternion(real_rotation_matrix)
        else:
            raise ValueError(
                "rotation_format must be 'euler', 'matrix', or 'quaternion'"
            )

        return new_rotation

    def get_xy_yaw(self):
        base_position = self.get_body_position()
        base_rotation_rpy = self.get_body_rotation()
        return np.array(
            [base_position[0], base_position[1], base_rotation_rpy[-1]]
        )

    def set_base_position(self, real_x_pos, real_y_pos, yaw):
        curr_base_pos = np.array(self.get_body_position())
        real_position = np.array([real_x_pos, real_y_pos, 0.0])

        curr_base_rot_rpy = self.get_body_rotation()
        real_rotation = np.array(
            [curr_base_rot_rpy[0], curr_base_rot_rpy[1], yaw]
        )
        sim_robot_pos, sim_robot_rot_matrix = transform_3d_coordinates(
            real_position, real_rotation, "real_to_sim", "matrix"
        )

        position = (
            sim_robot_pos
            - self.sim_obj.transformation.transform_vector(
                self.params.base_offset
            )
        )
        mn_matrix = mn.Matrix3(sim_robot_rot_matrix)

        # mn_matrix = mn.Matrix3(spot.get_rotation_matrix(real_robot_rot))
        target_trans = mn.Matrix4.from_(mn_matrix, mn.Vector3(*position))
        self.sim_obj.transformation = target_trans

    def set_base_velocity(self, x_vel, y_vel, ang_vel, vel_time):
        x_pos = x_vel * vel_time
        y_pos = y_vel * vel_time
        ang = ang_vel * vel_time
        return self.set_base_position(x_pos, y_pos, ang)

    def get_arm_joint_positions(
        self,
    ):
        return self.arm_joint_pos

    def set_arm_joint_positions(self, positions, format="degrees"):
        """
        Joint angles: sh0, sh1, hr1, el0, el1, wr0, wr1
        """
        if format == "degrees":
            positions = np.deg2rad(positions)
        self.arm_joint_pos = positions

    def transform_ee_rot(self, sim_rot):
        correction_R = euler_to_matrix(np.deg2rad([-90.0, 0.0, 90.0])).T

        real_rot = correction_R @ sim_rot

        # # Matrix to swap roll and yaw (90-degree rotations around Y axis)
        swap_matrix = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

        # # Combine both transformations
        real_rot = swap_matrix @ real_rot @ swap_matrix.T
        return real_rot

    def get_ee_global_pose(self):
        global_T_ee = self.ee_transform()
        real_global_pos = transform_position(global_T_ee.translation)
        real_global_rot = self.transform_ee_rot(global_T_ee.rotation())
        real_global_T_ee = create_tform_matrix(
            real_global_pos, real_global_rot, rotation_format="mn"
        )
        return real_global_T_ee

    def get_ee_local_pose_matrix(self):
        real_global_T_ee = self.get_ee_global_pose()
        real_global_T_body = self.global_T_body()

        return real_global_T_body.inverted() @ real_global_T_ee

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
        real_body_T_ee = self.get_ee_local_pose_matrix()
        real_ee_quat = matrix_to_quaternion(real_body_T_ee.rotation())
        return np.array(real_body_T_ee.translation), real_ee_quat

    def get_ee_pos_in_body_frame(self):
        real_body_T_ee = self.get_ee_local_pose_matrix()
        real_local_ee_pos = real_body_T_ee.translation
        real_local_ee_rpy = matrix_to_euler(real_body_T_ee.rotation())

        return real_local_ee_pos, real_local_ee_rpy

    def __init__(
        self, agent_cfg, sim, limit_robo_joints=True, fixed_base=True
    ):
        super().__init__(
            agent_cfg,
            sim,
            limit_robo_joints,
            fixed_base,
        )
