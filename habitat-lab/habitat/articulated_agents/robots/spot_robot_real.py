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

    def transform_to_real(self, sim_tform_matrix):
        # real_tform_matrix = self.xyz_T_hab(sim_tform_matrix)
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

    def set_arm_joint_positions(self, positions, format="radians"):
        """
        Joint angles: sh0, sh1, hr1, el0, el1, wr0, wr1
        """
        if format == "degrees":
            positions = np.deg2rad(positions)
        self.arm_joint_pos = positions

    def transform_ee_rot(self, sim_rot):
        correction_R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        real_rot = correction_R @ np.array(sim_rot)
        return real_rot

    def convert_matrix_habitat_to_standard(
        self, matrix: mn.Matrix4
    ) -> mn.Matrix4:
        """
        Convert a Matrix4 from Habitat convention (x=right, y=up, z=backwards)
        to standard convention (x=forward, y=left, z=up).

        Args:
            matrix: Magnum Matrix4 in Habitat convention

        Returns:
            Matrix4 in standard convention
        """
        # Convert Matrix4 to numpy array in row-major order for easier manipulation
        mat_np = np.array(
            [
                [matrix[0][0], matrix[1][0], matrix[2][0], matrix[3][0]],
                [matrix[0][1], matrix[1][1], matrix[2][1], matrix[3][1]],
                [matrix[0][2], matrix[1][2], matrix[2][2], matrix[3][2]],
                [matrix[0][3], matrix[1][3], matrix[2][3], matrix[3][3]],
            ]
        )

        # Create permutation matrix
        perm = np.array(
            [
                [0, 0, -1, 0],  # New x comes from negated old z
                [-1, 0, 0, 0],  # New y comes from negated old x
                [0, 1, 0, 0],  # New z comes from old y
                [0, 0, 0, 1],
            ]
        )

        # Apply the permutation
        result = mat_np.copy()
        result[:3, :3] = perm[:3, :3] @ mat_np[:3, :3] @ perm[:3, :3].T
        result[:3, 3] = perm[:3, :3] @ mat_np[:3, 3]

        return mn.Matrix4(result)

    @property
    def base_transformation_YZX(self):
        add_rot = mn.Matrix4.rotation(
            mn.Rad(-np.pi / 2), mn.Vector3(1.0, 0, 0)
        )
        return self.sim_obj.transformation @ add_rot

    @property
    def base_transformation(self):
        global_T_base_YZX = self.base_transformation_YZX
        global_T_base = self.convert_matrix_habitat_to_standard(
            global_T_base_YZX
        )
        print(
            "SIM global_T_base_YZX = ",
            global_T_base_YZX.translation,
            np.rad2deg(matrix_to_euler(global_T_base_YZX.rotation())),
        )
        print(
            "REAL global_T_base = ",
            global_T_base.translation,
            np.rad2deg(matrix_to_euler(global_T_base.rotation())),
        )
        # global_T_base_trans = transform_position(global_T_base_YZX.translation)
        # # base_xyz_T_hab_rot = mn.Matrix4(
        # #     [
        # #         [0.0, 1.0, 0.0, 0.0],
        # #         [0.0, 0.0, -1.0, 0.0],
        # #         [1.0, 0.0, 0.0, 0.0],
        # #         [0.0, 0.0, 0.0, 1.0],
        # #     ]
        # # ).transposed()
        # base_xyz_T_hab_rot = mn.Matrix4(
        #     [
        #         [0.0, -1.0, 0.0, 0.0],
        #         [-1.0, 0.0, 0.0, 0.0],
        #         [0.0, 0.0, 1.0, 0.0],
        #         [0.0, 0.0, 0.0, 1.0],
        #     ]
        # ).transposed()
        # global_T_base_rot = (global_T_base_YZX @ base_xyz_T_hab_rot).rotation()
        # global_T_base = mn.Matrix4.from_(
        #     global_T_base_rot, global_T_base_trans
        # )
        return global_T_base

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
        global_T_ee_YZX = self.ee_transform_YZX(ee_index)
        global_T_ee_trans = transform_position(global_T_ee_YZX.translation)
        ee_xyz_T_hab_rot = mn.Matrix4(
            [
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ).transposed()
        global_T_ee_rot = (global_T_ee_YZX @ ee_xyz_T_hab_rot).rotation()
        global_T_ee = mn.Matrix4.from_(global_T_ee_rot, global_T_ee_trans)
        return global_T_ee

    def get_ee_global_pose(self):
        global_T_ee = self.ee_transform_YZX()
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
        # real_ee_quat = matrix_to_quaternion(real_body_T_ee.rotation())
        real_ee_quat = quaternion.from_rotation_matrix(
            real_body_T_ee.rotation()
        )
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
