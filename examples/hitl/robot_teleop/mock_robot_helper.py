#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import magnum as mn
import numpy as np
from robot_camera_sensor_suite import RobotCameraSensorSuite

from scripts.robot import Robot


class MockRobotHelper:
    def __init__(self, sim):
        try:
            from murp.mock.mock_mobile_tmr_robot import MockMobileTMRRobot
        except ImportError as e:
            raise ImportError(
                f"Failed to import murp.mock.mock_mobile_tmr_robot. Did you install the `murp` package? See examples/hitl/robot_teleop/README.md Simulator Process. Raw import error: {e}"
            )
        self._murp_mock_robot = MockMobileTMRRobot(do_synthesize_images=False)
        self._hitl_robot: Optional[Robot] = None
        self._sim = sim

        self._robot_camera_sensor_suite: Optional[
            RobotCameraSensorSuite
        ] = None

    def set_hitl_robot(self, hitl_robot: Robot, robot_cfg):
        self._hitl_robot = hitl_robot

        if self._robot_camera_sensor_suite:
            self._robot_camera_sensor_suite.close()
        if "camera_sensors" in robot_cfg:
            self._robot_camera_sensor_suite = RobotCameraSensorSuite(
                self._sim, hitl_robot.ao, robot_cfg["camera_sensors"]
            )

    def draw_debug(self, dblr):
        if self._robot_camera_sensor_suite:
            self._robot_camera_sensor_suite.draw_debug(dblr)

    def update_pre_sim_step(self, dt):
        if not self._murp_mock_robot or not self._hitl_robot:
            return

        self._murp_mock_robot.poll_for_messages()

        if True:  # base linear and angular vel
            base_vel = self._murp_mock_robot.base.get_commanded_velocity()
            assert base_vel[1] == 0.0

            start = self._hitl_robot.ao.translation
            end = mn.Vector3(start)

            end = end + self._hitl_robot.ao.transformation.transform_vector(
                mn.Vector3(base_vel[0] * dt, 0, 0)
            )

            r = mn.Quaternion.rotation(
                mn.Rad(base_vel[2] * dt), mn.Vector3(0, 1, 0)
            )
            self._hitl_robot.ao.rotation = r * self._hitl_robot.ao.rotation

            if start != end:
                self._hitl_robot.ao.translation = (
                    self._sim.pathfinder.try_step(start, end)
                )

        motor_ids = []
        commanded_positions = np.array([], dtype=np.float32)

        # convention for murp: index, middle, pinky, thumb
        # convention for robot_settings.xml: should now be the same

        if self._hitl_robot.using_joint_motors:
            for hand_idx in range(2):
                joint_motor_lists = self._hitl_robot.pos_subsets[
                    "left_hand" if hand_idx == 0 else "right_hand"
                ].joint_motors
                for motor_list in joint_motor_lists:
                    assert len(motor_list) == 1
                    motor_ids.append(motor_list[0])
                commanded_positions = np.append(
                    commanded_positions,
                    (
                        self._murp_mock_robot.left_hand
                        if hand_idx == 0
                        else self._murp_mock_robot.right_hand
                    ).commanded_positions,
                )
                assert len(motor_ids) == len(commanded_positions)

            for arm_idx in range(2):
                joint_motor_lists = self._hitl_robot.pos_subsets[
                    "left_arm" if arm_idx == 0 else "right_arm"
                ].joint_motors
                for motor_list in joint_motor_lists:
                    assert len(motor_list) == 1
                    motor_ids.append(motor_list[0])
                commanded_positions = np.append(
                    commanded_positions,
                    (
                        self._murp_mock_robot.left_arm
                        if arm_idx == 0
                        else self._murp_mock_robot.right_arm
                    ).get_target_joint_positions(),
                )
                assert len(motor_ids) == len(commanded_positions)

            for motor_id, commanded_pos in zip(motor_ids, commanded_positions):
                jms = self._hitl_robot.ao.get_joint_motor_settings(motor_id)
                jms.position_target = commanded_pos
                self._hitl_robot.ao.update_joint_motor(motor_id, jms)
        else:
            # directly set joint positions
            curr_robot_joint_positions = self._hitl_robot.ao.joint_positions

            for hand_idx in range(2):
                commanded_positions = (
                    self._murp_mock_robot.left_hand
                    if hand_idx == 0
                    else self._murp_mock_robot.right_hand
                ).commanded_positions
                link_ixs = self._hitl_robot.pos_subsets[
                    "left_hand" if hand_idx == 0 else "right_hand"
                ].link_ixs
                for i, link_ix in enumerate(link_ixs):
                    dof = self._hitl_robot.ao.get_link_joint_pos_offset(
                        link_ix
                    )
                    curr_robot_joint_positions[dof] = commanded_positions[i]

            for arm_idx in range(2):
                commanded_positions = (
                    self._murp_mock_robot.left_arm
                    if arm_idx == 0
                    else self._murp_mock_robot.right_arm
                ).get_target_joint_positions()
                link_ixs = self._hitl_robot.pos_subsets[
                    "left_arm" if arm_idx == 0 else "right_arm"
                ].link_ixs
                for i, link_ix in enumerate(link_ixs):
                    dof = self._hitl_robot.ao.get_link_joint_pos_offset(
                        link_ix
                    )
                    curr_robot_joint_positions[dof] = commanded_positions[i]

            self._hitl_robot.ao.joint_positions = curr_robot_joint_positions

    def update_post_sim_step(self, post_sim_update_dict):
        if not self._murp_mock_robot or not self._hitl_robot:
            return

        if True:  # base
            base_xyz = self._hitl_robot.ao.translation

            mat = self._hitl_robot.ao.transformation
            yaw = np.arctan2(mat[2][0], mat[0][0])
            self._murp_mock_robot.base.set_pose(base_xyz.x, base_xyz.z, yaw)

        curr_robot_joint_positions = self._hitl_robot.ao.joint_positions

        for hand_idx in range(2):
            link_ixs = self._hitl_robot.pos_subsets[
                "left_hand" if hand_idx == 0 else "right_hand"
            ].link_ixs
            curr_joint_positions = np.zeros(len(link_ixs), dtype=np.float32)
            for i, link_ix in enumerate(link_ixs):
                dof = self._hitl_robot.ao.get_link_joint_pos_offset(link_ix)
                curr_joint_positions[i] = curr_robot_joint_positions[dof]
            mock_hand = (
                self._murp_mock_robot.left_hand
                if hand_idx == 0
                else self._murp_mock_robot.right_hand
            )
            mock_hand.set_joint_state(curr_joint_positions)

        for arm_idx in range(2):
            link_ixs = self._hitl_robot.pos_subsets[
                "left_arm" if arm_idx == 0 else "right_arm"
            ].link_ixs
            curr_joint_positions = np.zeros(len(link_ixs), dtype=np.float32)
            for i, link_ix in enumerate(link_ixs):
                dof = self._hitl_robot.ao.get_link_joint_pos_offset(link_ix)
                curr_joint_positions[i] = curr_robot_joint_positions[dof]
            mock_arm = (
                self._murp_mock_robot.left_arm
                if arm_idx == 0
                else self._murp_mock_robot.right_arm
            )
            mock_arm.set_current_joint_positions(curr_joint_positions)

        self._murp_mock_robot.publish_proprioception_state()

        self._robot_camera_sensor_suite.draw_and_publish_observations(
            self._murp_mock_robot.camera_suite
        )
