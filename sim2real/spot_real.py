# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# mypy: ignore-errors
# Copyright (c) 2021 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

""" Easy-to-use wrapper for properly controlling Spot """
import os
import os.path as osp
import time
import traceback
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import bosdyn.client.lease
import bosdyn.client.util
import cv2
import magnum as mn
import numpy as np
import quaternion
import rospy

try:
    import sophuspy as sp
except Exception as e:
    print(f"Cannot import sophuspy due to {e}. Import sophus instead")
    import sophus as sp

from bosdyn.api import arm_command_pb2
from bosdyn.api.geometry_pb2 import SE2Velocity, SE2VelocityLimit, Vec2, Vec3
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import (
    VISION_FRAME_NAME,
    get_a_tform_b,
    get_vision_tform_body,
)
from google.protobuf import wrappers_pb2  # type: ignore


class Spot:
    def __init__(self, client_name_prefix):
        pass

    def open_gripper(self):
        """Does not block, be careful!"""
        gripper_command = RobotCommandBuilder.claw_gripper_open_command()
        self.command_client.robot_command(gripper_command)

    def close_gripper(self):
        """Does not block, be careful!"""
        gripper_command = RobotCommandBuilder.claw_gripper_close_command()
        self.command_client.robot_command(gripper_command)

    def set_base_velocity(
        self,
        x_vel,
        y_vel,
        ang_vel,
        vel_time,
        disable_obstacle_avoidance=False,
        return_cmd=False,
    ):
        body_tform_goal = math_helpers.SE2Velocity(
            x=x_vel, y=y_vel, angular=ang_vel
        )
        params = spot_command_pb2.MobilityParams(
            obstacle_params=spot_command_pb2.ObstacleParams(
                disable_vision_body_obstacle_avoidance=disable_obstacle_avoidance,
                disable_vision_foot_obstacle_avoidance=False,
                disable_vision_foot_constraint_avoidance=False,
                obstacle_avoidance_padding=0.05,  # in meters
            )
        )
        command = RobotCommandBuilder.synchro_velocity_command(
            v_x=body_tform_goal.linear_velocity_x,
            v_y=body_tform_goal.linear_velocity_y,
            v_rot=body_tform_goal.angular_velocity,
            params=params,
        )

        if return_cmd:
            return command

        cmd_id = self.command_client.robot_command(
            command, end_time_secs=time.time() + vel_time
        )

        return cmd_id

    def set_base_position(
        self,
        x_pos,
        y_pos,
        yaw,
        end_time,
        relative=False,
        max_fwd_vel=2,
        max_hor_vel=2,
        max_ang_vel=np.pi / 2,
        disable_obstacle_avoidance=False,
        blocking=False,
    ):
        vel_limit = SE2VelocityLimit(
            max_vel=SE2Velocity(
                linear=Vec2(x=max_fwd_vel, y=max_hor_vel), angular=max_ang_vel
            ),
            min_vel=SE2Velocity(
                linear=Vec2(x=-max_fwd_vel, y=-max_hor_vel),
                angular=-max_ang_vel,
            ),
        )
        params = spot_command_pb2.MobilityParams(
            vel_limit=vel_limit,
            obstacle_params=spot_command_pb2.ObstacleParams(
                disable_vision_body_obstacle_avoidance=disable_obstacle_avoidance,
                disable_vision_foot_obstacle_avoidance=False,
                disable_vision_foot_constraint_avoidance=False,
                obstacle_avoidance_padding=0.05,  # in meters
            ),
        )
        if relative:
            global_x_pos, global_y_pos, global_yaw = (
                self.get_global_from_local(x_pos, y_pos, yaw)
            )
        else:
            global_x_pos, global_y_pos, global_yaw = (
                self.xy_yaw_home_to_global(x_pos, y_pos, yaw)
            )
        robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=global_x_pos,
            goal_y=global_y_pos,
            goal_heading=global_yaw,
            frame_name=VISION_FRAME_NAME,
            params=params,
        )
        cmd_id = self.command_client.robot_command(
            robot_cmd, end_time_secs=time.time() + end_time
        )

        if blocking:
            cmd_status = None
            while cmd_status != 1:
                time.sleep(0.1)
                feedback_resp = self.get_cmd_feedback(cmd_id)
                cmd_status = (
                    feedback_resp.feedback.synchronized_feedback
                ).mobility_command_feedback.se2_trajectory_feedback.status
            return None

        return cmd_id

    def get_robot_state(self):
        return self.robot_state_client.get_robot_state()

    def get_arm_proprioception(self, robot_state=None):
        """Return state of each of the 6 joints of the arm"""
        if robot_state is None:
            robot_state = self.robot_state_client.get_robot_state()
        arm_joint_states = OrderedDict(
            {
                i.name[len("arm0.") :]: i
                for i in robot_state.kinematic_state.joint_states
                if i.name in ARM_6DOF_NAMES
            }
        )

        return arm_joint_states

    def get_arm_joint_positions(self, as_array=True):
        """
        Gives in joint positions of the arm in radians in the following order
        Ordering: sh0, sh1, el0, el1, wr0, wr1
        :param as_array: bool, True for output as an np.array, False for list
        :return: 6 element data structure (np.array or list) of joint positions as radians
        """
        arm_joint_states = self.get_arm_proprioception()
        arm_joint_positions = np.fromiter(
            (
                arm_joint_states[joint].position.value
                for joint in arm_joint_states
            ),
            float,
        )

        if as_array:
            return arm_joint_positions
        return arm_joint_positions.tolist()

    def set_arm_joint_positions(
        self,
        positions,
        travel_time=1.0,
        max_vel=2.5,
        max_acc=15,
        return_cmd=False,
    ):
        """
        Takes in 6 joint targets and moves each arm joint to the corresponding target.
        Ordering: sh0, sh1, el0, el1, wr0, wr1
        :param positions: np.array or list of radians
        :param travel_time: how long execution should take
        :param max_vel: max allowable velocity
        :param max_acc: max allowable acceleration
        :return: cmd_id
        """
        sh0, sh1, el0, el1, wr0, wr1 = positions
        traj_point = RobotCommandBuilder.create_arm_joint_trajectory_point(
            sh0, sh1, el0, el1, wr0, wr1, travel_time
        )
        arm_joint_traj = arm_command_pb2.ArmJointTrajectory(
            points=[traj_point],
            maximum_velocity=wrappers_pb2.DoubleValue(value=max_vel),
            maximum_acceleration=wrappers_pb2.DoubleValue(value=max_acc),
        )
        command = make_robot_command(arm_joint_traj)

        if return_cmd:
            return command

        cmd_id = self.command_client.robot_command(command)

        return cmd_id

    def set_base_vel_and_arm_pos(
        self,
        x_vel,
        y_vel,
        ang_vel,
        arm_positions,
        travel_time,
        disable_obstacle_avoidance=False,
    ):
        base_cmd = self.set_base_velocity(
            x_vel,
            y_vel,
            ang_vel,
            vel_time=travel_time,
            disable_obstacle_avoidance=disable_obstacle_avoidance,
            return_cmd=True,
        )
        arm_cmd = self.set_arm_joint_positions(
            arm_positions, travel_time=travel_time, return_cmd=True
        )
        synchro_command = RobotCommandBuilder.build_synchro_command(
            base_cmd, arm_cmd
        )
        cmd_id = self.command_client.robot_command(
            synchro_command, end_time_secs=time.time() + travel_time
        )
        return cmd_id

    def set_base_vel_and_arm_ee_pos(
        self,
        x_vel,
        y_vel,
        ang_vel,
        arm_ee_action,
        travel_time,
        disable_obstacle_avoidance=False,
    ):
        base_cmd = self.set_base_velocity(
            x_vel,
            y_vel,
            ang_vel,
            vel_time=travel_time,
            disable_obstacle_avoidance=disable_obstacle_avoidance,
            return_cmd=True,
        )
        arm_cmd = self.move_gripper_to_point(
            point=arm_ee_action[0:3],
            rotation=list(arm_ee_action[3:]),
            seconds_to_goal=travel_time,
            return_cmd=True,
        )
        synchro_command = RobotCommandBuilder.build_synchro_command(
            base_cmd, arm_cmd
        )
        cmd_id = self.command_client.robot_command(
            synchro_command, end_time_secs=time.time() + travel_time
        )
        return cmd_id

    def set_arm_ee_pos(
        self,
        arm_ee_action,
        travel_time,
    ):
        arm_cmd = self.move_gripper_to_point(
            point=arm_ee_action[0:3],
            rotation=list(arm_ee_action[3:]),
            seconds_to_goal=travel_time,
            return_cmd=True,
        )
        synchro_command = RobotCommandBuilder.build_synchro_command(arm_cmd)
        cmd_id = self.command_client.robot_command(
            synchro_command, end_time_secs=time.time() + travel_time
        )
        return cmd_id

    def get_xy_yaw(self, use_boot_origin=False, robot_state=None):
        """
        Returns the relative x and y distance from start, as well as relative heading
        """
        if robot_state is None:
            robot_state = self.robot_state_client.get_robot_state()
        robot_state_kin = robot_state.kinematic_state
        self.body = get_vision_tform_body(robot_state_kin.transforms_snapshot)
        robot_tform = self.body
        yaw = math_helpers.quat_to_eulerZYX(robot_tform.rotation)[0]
        if self.global_T_home is None or use_boot_origin:
            return robot_tform.x, robot_tform.y, yaw
        return self.xy_yaw_global_to_home(robot_tform.x, robot_tform.y, yaw)

    def get_ee_transform(self, frame_name: str = "body"):
        """
        Get ee transformation from base (body) to hand frame
        """
        body_T_hand = get_a_tform_b(
            self.robot_state_client.get_robot_state().kinematic_state.transforms_snapshot,
            frame_name,
            "hand",
        )
        return body_T_hand

    def get_ee_pos_in_body_frame(self, frame_name: str = "body"):
        """
        Return ee xyz position and roll, pitch, yaw
        """
        # Get transformation
        body_T_hand = self.get_ee_transform(frame_name)

        # Get rotation. BD API returns values with the order of yaw, pitch, roll.
        theta = math_helpers.quat_to_eulerZYX(body_T_hand.rotation)
        # Change the order to roll, pitch, yaw
        theta = np.array(theta)[::-1]

        # Get position x,y,z
        position = (
            self.robot_state_client.get_robot_state()
            .kinematic_state.transforms_snapshot.child_to_parent_edge_map[
                "hand"
            ]
            .parent_tform_child.position
        )

        return np.array([position.x, position.y, position.z]), theta
