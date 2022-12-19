#!/usr/bin/env python


"""
Based on code from Willow Garage:
    https://github.com/ros-visualization/visualization_tutorials/blob/indigo-devel/interactive_marker_tutorials/scripts/basic_controls.py

"""


import numpy as np
import rospy
import threading

from interactive_markers.interactive_marker_server import *
from interactive_markers.menu_handler import *
from visualization_msgs.msg import *
from geometry_msgs.msg import Pose, Point, Quaternion

from home_robot.hardware.stretch_ros import HelloStretchROSInterface
from home_robot.agent.motion.robot import (
    STRETCH_HOME_Q,
    STRETCH_GRASP_FRAME,
    STRETCH_TO_GRASP,
)
from home_robot.agent.motion.robot import HelloStretchIdx
from home_robot.hw.ros.utils import *
from home_robot.hw.ros.position_mode import SwitchToPositionMode


"""
Copyright (c) 2011, Willow Garage, Inc.
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Willow Garage, Inc. nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES LOSS OF USE, DATA, OR PROFITS OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import rospy
import copy

from interactive_markers.interactive_marker_server import *
from interactive_markers.menu_handler import *
from visualization_msgs.msg import *
from geometry_msgs.msg import Point
from tf.broadcaster import TransformBroadcaster

from random import random
from math import sin

server = None
br = None
counter = 0

root_link = "map"


def makeBox(msg, r, g, b):
    marker = Marker()

    marker.type = Marker.CUBE
    marker.scale.x = msg.scale * 0.45 * 0.25
    marker.scale.y = msg.scale * 0.45 * 0.25
    marker.scale.z = msg.scale * 0.45 * 0.25
    marker.color.r = r
    marker.color.g = g
    marker.color.b = b
    marker.color.a = 0.75

    return marker


def makeBoxControl(msg, r, g, b):
    """from willow teleop code"""
    control = InteractiveMarkerControl()
    control.always_visible = True
    control.markers.append(makeBox(msg, r, g, b))
    msg.controls.append(control)
    return control


#####################################################################
# Marker Creation


def makeBaseMarker(position, orientation, root_link):
    int_marker = InteractiveMarker()
    int_marker.header.frame_id = root_link
    int_marker.pose.position = position
    int_marker.pose.orientation = orientation
    int_marker.scale = 0.5
    int_marker.name = "base_motion"
    int_marker.description = ""  # base motions in plane"

    # Add a box
    makeBoxControl(int_marker, 0.0, 0.0, 1.0)
    int_marker.controls[0].interaction_mode = InteractiveMarkerControl.FIXED

    # Add Motion
    control = InteractiveMarkerControl()
    control.orientation.w = 1
    control.orientation.x = 1
    control.orientation.y = 0
    control.orientation.z = 0
    control.name = "move_x"
    control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
    int_marker.controls.append(control)

    # Add Z rotation
    control = InteractiveMarkerControl()
    control.orientation.w = 1
    control.orientation.x = 0
    control.orientation.y = 1
    control.orientation.z = 0
    control.name = "rotate_z"
    control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
    int_marker.controls.append(control)
    return int_marker


class TeleopMarkerServer(object):
    """server that publishes interactive marker messages and owns a robot interface"""

    def __init__(self, connect_to_robot=True):
        self.marker_server = InteractiveMarkerServer("stretch_controls")
        self.root_link = "base_link"
        self.switch_to_position_mode = SwitchToPositionMode()
        if connect_to_robot:
            # Connect to the robot for teleoperation
            self.robot = HelloStretchROSInterface(visualize_planner=False)
            # Model for executing planning queries
            self.model = self.robot.get_model()
        else:
            # Dummy node, will not execute anything
            self.robot = None
            self.model = None

        self._lock = threading.Lock()

        # Create a menu handler
        self.menu_handler = MenuHandler()
        # menu_handler.insert( "Home the robot", callback=self_cb_home())
        self.menu_handler.insert("Stow the arm", callback=self._cb_stow)
        self.menu_handler.insert("Raise the arm", callback=self._cb_raise)
        self.menu_handler.insert("Look straight", callback=self._cb_look_straight)
        self.menu_handler.insert("Look forward", callback=self._cb_look_front)
        self.menu_handler.insert("Look at gripper", callback=self._cb_look_at_ee)
        self.menu_handler.insert("Open gripper", callback=self._cb_open_ee)
        self.menu_handler.insert("Close gripper", callback=self._cb_close_ee)

        # create a timer to update the published transforms
        self.counter = 0
        self.done = False
        self.br = TransformBroadcaster()
        self.add_base_marker()
        self.add_arm_marker()
        self.add_gripper_marker()
        rospy.Timer(rospy.Duration(0.01), self._cb_update_poses)
        self.marker_server.applyChanges()

    def _cb_open_ee(self, msg):
        print("Opening the gripper")
        with self._lock:
            q, _ = self.robot.update()
            q = self.model.update_gripper(q, open=True)
            self.robot.goto(q, move_base=False, wait=False)

    def _cb_close_ee(self, msg):
        print("Closing the gripper")
        with self._lock:
            q, _ = self.robot.update()
            q = self.model.update_gripper(q, open=False)
            self.robot.goto(q, move_base=False, wait=False)

    def _cb_stow(self, msg):
        print("Stowing the robot arm")
        with self._lock:
            q, _ = self.robot.update()
            q[HelloStretchIdx.ARM] = 0
            # q = self.model.update_look_front(q)
            self.robot.goto(q, move_base=False, wait=True)
            self.robot.goto(STRETCH_HOME_Q, move_base=False, wait=False)

    def _cb_raise(self, msg):
        print("Raising the robot arm")
        with self._lock:
            q, _ = self.robot.update()
            q[HelloStretchIdx.ARM] = 0
            q[HelloStretchIdx.LIFT] = 1.1
            # q = self.model.update_look_front(q)
            self.robot.goto(q, move_base=False, wait=True)
            q = self.model.update_look_at_ee(q)
            self.robot.goto(q, move_base=False, wait=False)

    def _cb_look_at_ee(self, msg):
        print("Looking at the arm ee")
        with self._lock:
            q, _ = self.robot.update()
            q = self.model.update_look_at_ee(q)
            self.robot.goto(q, move_base=False, wait=False)

    def _cb_look_straight(self, msg):
        print("Looking straight ahead")
        with self._lock:
            q, _ = self.robot.update()
            q = self.model.update_look_ahead(q)
            self.robot.goto(q, move_base=False, wait=False)

    def _cb_look_front(self, msg):
        print("Looking at the front")
        with self._lock:
            q, _ = self.robot.update()
            q = self.model.update_look_front(q)
            self.robot.goto(q, move_base=False, wait=False)

    def _cb_send_arm_command(self, feedback):
        arm_pose = self.robot.get_pose("link_arm_l0")
        with self._lock:
            q0, _ = self.robot.update()
        arm_pose[:3, 3] = np.array(
            [
                feedback.pose.position.x,
                feedback.pose.position.y,
                feedback.pose.position.z,
            ]
        )
        q = self.model.lift_arm_ik_from_matrix(arm_pose, q0)
        if q is not None:
            q[HelloStretchIdx.HEAD_PAN] = q0[HelloStretchIdx.HEAD_PAN]
            q[HelloStretchIdx.HEAD_TILT] = q0[HelloStretchIdx.HEAD_TILT]
            q[HelloStretchIdx.GRIPPER] = q0[HelloStretchIdx.GRIPPER]
            self.robot.goto(q, move_base=False, wait=False)

    def _cb_send_gripper_command(self, feedback):
        with self._lock:
            q, _ = self.robot.update()
        ranges = self.model.range[
            HelloStretchIdx.WRIST_ROLL : (HelloStretchIdx.WRIST_YAW + 1)
        ]
        rpy = tra.euler_from_matrix(matrix_from_pose_msg(feedback.pose))
        rpy = np.clip(rpy, ranges[:, 0], ranges[:, 1])
        q[HelloStretchIdx.WRIST_ROLL : (HelloStretchIdx.WRIST_YAW + 1)] = rpy
        self.robot.goto(q, move_base=False, wait=False)

    def add_arm_marker(self):
        """add the arm marker widget"""
        with self._lock:
            q, _ = self.robot.update()

        int_marker = self.make_arm_marker()
        self.arm_marker_name = int_marker.name
        self.marker_server.insert(int_marker, self._cb_send_arm_command)
        self.marker_server.setCallback(
            self.arm_marker_name,
            self._cb_send_arm_command,
            InteractiveMarkerFeedback.POSE_UPDATE,
        )
        self.menu_handler.apply(self.marker_server, self.base_marker_name)

    def add_base_marker(self):
        with self._lock:
            q, _ = self.robot.update()
        position = Point(q[0], q[1], 0.2)
        orientation = theta_to_quaternion_msg(q[2])
        int_marker = makeBaseMarker(position, orientation, root_link)
        self.base_marker_name = int_marker.name
        self.marker_server.insert(int_marker, self._cb_send_command)
        self.marker_server.setCallback(
            self.base_marker_name,
            self._cb_send_command,
            InteractiveMarkerFeedback.POSE_UPDATE,
        )
        self.menu_handler.apply(self.marker_server, self.base_marker_name)

    def get_x_cmd(self, cmd_xy, q):
        cur_xy = np.array([q[0], q[1]])
        dist = np.linalg.norm(cmd_xy - cur_xy)
        theta = np.arctan2(cmd_xy[0] - cur_xy[0], cmd_xy[1] - cur_xy[1])
        dirn = -1 if np.abs(theta - (np.pi / 2)) > 0 else 1
        return dist, dirn

    def make_arm_marker(self):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self.root_link
        arm_pose = self.robot.get_pose("link_arm_l0")
        x, y, z = arm_pose[:3, 3]
        int_marker.pose.position.x = x
        int_marker.pose.position.y = y
        int_marker.pose.position.z = z
        # int_marker.pose.orientation = orientation
        int_marker.scale = 0.25
        int_marker.name = "arm_control"
        int_marker.description = ""  # base motions in plane"

        # Add a box
        makeBoxControl(int_marker, 0.0, 1.0, 0.0)
        int_marker.controls[0].interaction_mode = InteractiveMarkerControl.FIXED

        # Add Y Motion
        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 0
        control.orientation.z = 1
        control.name = "move_y"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        # Add Z Motion
        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 1
        control.orientation.z = 0
        control.name = "move_z"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        return int_marker

    def add_gripper_marker(self):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self.root_link
        arm_pose = self.robot.get_pose("link_arm_l0")
        gripper_pose = self.robot.get_pose("link_straight_gripper")
        x, y, z = arm_pose[:3, 3]
        int_marker.pose.position.x = x
        int_marker.pose.position.y = y
        int_marker.pose.position.z = z + 0.2
        # int_marker.pose.orientation = orientation
        int_marker.scale = 0.25
        int_marker.name = "gripper"
        int_marker.description = ""  # base motions in plane"

        # Add a box
        makeBoxControl(int_marker, 1.0, 0.0, 0.0)
        int_marker.controls[0].interaction_mode = InteractiveMarkerControl.FIXED

        # Add X rotation
        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 1
        control.orientation.y = 0
        control.orientation.z = 0
        control.name = "rotate_x"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        # Add Y rotation
        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 0
        control.orientation.z = 1
        control.name = "rotate_y"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        # Add Z rotation
        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 1
        control.orientation.z = 0
        control.name = "rotate_z"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        self.gripper_marker_name = int_marker.name
        self.marker_server.insert(int_marker, self._cb_send_gripper_command)
        self.marker_server.setCallback(
            self.gripper_marker_name,
            self._cb_send_gripper_command,
            InteractiveMarkerFeedback.POSE_UPDATE,
        )

    def check_switch_to_position_mode(self):
        """only switch if necessary"""
        if not self.robot.in_position_mode():
            self.switch_to_position_mode()

    def _cb_send_command(self, feedback):
        x_tol = 0.01
        theta_tol = 0.01
        pose = feedback.pose
        T = matrix_from_pose_msg(feedback.pose)
        # T = np.linalg.inv(Tinv)
        Tinv = np.linalg.inv(T)
        with self._lock:
            q, _ = self.robot.update()

        # Get angle command
        R = tra.euler_matrix(0, 0, q[2])
        # R_inv = Tinv[:3, :3]
        R_dif = R @ Tinv
        angles = tra.euler_from_matrix(R_dif)
        print("------------")
        print("angles =", angles)
        cmd_theta = angles[2]

        # Get positional command
        cmd_xy = T[:2, 3]
        dist, dirn = self.get_x_cmd(cmd_xy, q)
        print("theta: robot at", q[2], "vs", cmd_theta)
        print("cmd xy =", cmd_xy)
        print("cur cy =", q[:2])
        print("dist =", dist)
        print(dist, dirn, cmd_theta)

        # Switch to positoin mode if we need to, before sending any commands
        self.check_switch_to_position_mode()

        # Correct and command
        dist2 = min(dist, 0.3)
        theta = np.clip(cmd_theta, -0.1, 0.1)
        if np.abs(cmd_theta) > 0.1 * dist:
            print("sending rotation command")
            if np.abs(theta) > theta_tol:
                self.robot.goto_theta(-1 * theta)
        else:
            if dist > x_tol and dist > np.abs(theta):
                self.robot.goto_x(-1 * dirn * dist2)
        self._reset(q)

    def get_rpy(self, q):
        roll = q[HelloStretchIdx.WRIST_ROLL]
        pitch = q[HelloStretchIdx.WRIST_PITCH]
        yaw = q[HelloStretchIdx.WRIST_YAW]
        return roll, pitch, yaw

    def _reset(self, q):
        # Reset everything to zero
        pose = Pose()
        # print(q[:2])
        T = tra.euler_matrix(0, 0, q[2])
        T[:2, 3] = q[:2]
        _, _, theta = tra.euler_from_matrix(T)
        pose.position.x = T[0, 3]
        pose.position.y = T[1, 3]
        pose.position.z = 0.2
        pose.orientation = theta_to_quaternion_msg(theta)
        self.marker_server.setPose(self.base_marker_name, pose)

        # Create a new pose marker for the end effector control
        pose2 = Pose()
        # Lookup a pose for the robot gripper
        arm_pose = self.robot.get_pose("link_arm_l0")
        x, y, z = arm_pose[:3, 3]
        pose2.position.x = x
        pose2.position.y = y
        pose2.position.z = z
        self.marker_server.setPose(self.arm_marker_name, pose2)

        # gripper_pose = self.robot.get_pose("link_gripper")
        # x, y, z = gripper_pose[:3, 3]
        pose3 = Pose()
        pos, rot = self.model.fk(q)
        qx, qy, qz, qw = rot
        ee_pose = tra.quaternion_matrix([qw, qx, qy, qz])
        ee_pose[:3, 3] = pos
        ee_pose = ee_pose @ STRETCH_TO_GRASP
        ex, ey, ez = ee_pose[:3, 3]
        roll, pitch, yaw = self.get_rpy(q)
        R = tra.euler_matrix(roll, pitch, yaw)
        # R = R @ tra.euler_matrix(-np.pi/2, 0, 0)
        pose3.position.x = ex
        pose3.position.y = ey
        pose3.position.z = ez
        # w, x, y, z = tra.quaternion_from_matrix(R)
        # pose3.orientation.x = x
        # pose3.orientation.y = y
        # pose3.orientation.z = z
        # pose3.orientation.w = w
        self.marker_server.setPose(self.gripper_marker_name, pose3)

        # Finish
        self.marker_server.applyChanges()

    def _cb_update_poses(self, msg):
        with self._lock:
            q, _ = self.robot.update()
        self._reset(q)
        self.counter += 1


if __name__ == "__main__":
    rospy.init_node("basic_controls")
    teleop_server = TeleopMarkerServer()
    rospy.spin()
