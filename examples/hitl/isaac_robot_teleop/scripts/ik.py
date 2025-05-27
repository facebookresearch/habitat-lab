#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ros_pymomentum.srv import InverseKinematics
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
import rclpy
from rclpy.node import Node
import numpy as np

from typing import Optional


import roboticstoolbox as rtb
from spatialmath import SE3, Quaternion
import magnum as mn

def to_ik_pose(
    legacy_tuple: Optional[tuple[mn.Vector3, mn.Quaternion]]
) -> str:
    """
    Converts a transformation into the expected type (spatialmath) and conventions for the IK solver.
    """

    if legacy_tuple is None:
        return None

    v = legacy_tuple[0]
    q = legacy_tuple[1]

    if v is None or q is None:
        return None

    # apply a corrective rotation to the local frame in order to align the palm
    r = mn.Quaternion.rotation(-mn.Rad(0), mn.Vector3(0, 0, 1))
    q = q * r

    R = (
        (Quaternion([q.scalar, q.vector[0], q.vector[1], q.vector[2]]))
        .unit()
        .SE3()
    )
    t = SE3([v.x, v.y, v.z])
    pose = t * R

    return pose


rclpy.init(args=None)

class DifferentialInverseKinematics(Node):

    def __init__(self):
        super().__init__('inverse_kinematics_client')
        self.cli = self.create_client(InverseKinematics, 'pymomentum_ik')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = InverseKinematics.Request()
        print("IK client initialized")
        self.robot = rtb.models.Panda()


    def get_ee_T(self):
        """
        Get the end effector transform in robot base space as a Matrix4.
        """
        return mn.Matrix4(self.robot.fkine(self.robot.q, end=self.robot.links[7]).A)

    def inverse_kinematics(self, pose, current_joint_positions):

        _pose = Pose()
        _pose.position.x = pose.x
        _pose.position.y = pose.y
        _pose.position.z = pose.z
        _pose.orientation.x = pose.UnitQuaternion().A[1]
        _pose.orientation.y = pose.UnitQuaternion().A[2]
        _pose.orientation.z = pose.UnitQuaternion().A[3]
        _pose.orientation.w = pose.UnitQuaternion().A[0]

        _current_joint_positions = JointState()
        _current_joint_positions.position = current_joint_positions


        self.req.end_effector_pose = _pose
        self.req.current_joint_positions = _current_joint_positions
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        response = self.future.result()

        self.robot.q = np.array(response.desired_joint_positions.position)
        return self.robot.q

    