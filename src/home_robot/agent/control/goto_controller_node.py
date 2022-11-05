from typing import List, Optional
import threading

import numpy as np
import rospy
from geometry_msgs.msg import Twist, PoseStamped

from home_robot.agent.control.diff_drive_vel_control import DiffDriveVelocityControl
from home_robot.utils.geometry import xyt_global_to_base
from home_robot.utils.ros_geometry import pose_ros2sp


CONTROL_HZ = 20


class GotoVelocityController:
    """
    Self-contained controller module for moving a diff drive robot to a target goal.
    Target goal is update-able at any given instant.
    """

    def __init__(
        self,
        hz: float,
    ):
        self.hz = hz
        self.dt = 1.0 / self.hz

        # Control module
        self.control = DiffDriveVelocityControl(hz)

        # Publishers
        rospy.init_node("goto_controller")
        self._vel_command_pub = rospy.Publisher("/stretch/cmd_vel", Twist, queue_size=1)

        # Initialize
        self.xyt_loc = np.zeros(3)
        self.xyt_goal = self.xyt_loc
        self.track_yaw = True

    def _pose_update_callback(self, pose):
        pose_sp = pose_ros2sp(pose.pose)
        self.xyt_loc = np.array(
            [pose_sp.translation()[0], pose_sp.translation()[1], pose_sp.so3().log()[2]]
        )

    def _compute_error_pose(self):
        """
        Updates error based on robot localization
        """
        xyt_err = xyt_global_to_base(self.xyt_goal, self.xyt_loc)
        if not self.track_yaw:
            xyt_err[2] = 0.0

        return xyt_err

    def _set_velocity(self, v_m, w_r):
        cmd = Twist()
        cmd.linear.x = v_m
        cmd.angular.z = w_r
        self._vel_command_pub.publish(cmd)

    def _run_control_loop(self):
        rate = rospy.Rate(self.hz)

        while True:
            # Get state estimation
            xyt_err = self._compute_error_pose()

            # Compute control
            v_cmd, w_cmd = self.control(xyt_err)

            # Command robot
            self._set_velocity(v_cmd, w_cmd)

            # Spin
            rate.sleep()

    def set_goal(
        self,
        xyt_position: List[float],
    ):
        self.xyt_goal = xyt_position

    def enable_yaw_tracking(self, value: bool = True):
        self.track_yaw = value

    def check_at_goal(self) -> bool:
        xyt_err = self._compute_error_pose()

        xy_fulfilled = np.linalg.norm(xyt_err[0:2]) <= self.lin_error_tol

        t_fulfilled = True
        if self.track_yaw:
            t_fulfilled = abs(xyt_err[2]) <= self.ang_error_tol

        return xy_fulfilled and t_fulfilled

    def main(self):
        # Subscribers
        rospy.Subscriber(
            "/state_estimator", PoseStamped, self._pose_update_callback, queue_size=1
        )

        # Services (Why is it so hard to provide a service in ROS?)
        rospy.Service("set_goal", TODO_SERVICE_TYPE, self.set_goal)
        rospy.Service("check_at_goal", TODO_SERVICE_TYPE, self.check_at_goal)
        rospy.Service(
            "enable_yaw_tracking", TODO_SERVICE_TYPE, self.enable_yaw_tracking
        )

        # Run controller
        self._run_control_loop()


if __name__ == "__main__":
    node = GotoVelocityController(CONTROL_HZ)
    node.main()
