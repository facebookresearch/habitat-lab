from typing import List, Optional
import threading

import numpy as np
import rospy
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist, Pose, PoseStamped

from home_robot.agent.control.diff_drive_vel_control import DiffDriveVelocityControl
from home_robot.utils.geometry import xyt_global_to_base, sophus2xyt
from home_robot.utils.geometry.ros import pose_ros2sophus


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
        self.xyt_goal = None
        self.track_yaw = True

    def _pose_update_callback(self, msg: PoseStamped):
        pose_sp = pose_ros2sophus(msg.pose)
        self.xyt_loc = sophus2xyt(pose_sp)

    def _goal_update_callback(self, msg: Pose):
        pose_sp = pose_ros2sophus(msg)
        self.xyt_goal = sophus2xyt(pose_sp)

    def _yaw_toggle_callback(self, msg: Bool):
        self.track_yaw = msg.data

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
            if self.xyt_goal is not None:
                # Get state estimation
                xyt_err = self._compute_error_pose()

                # Compute control
                v_cmd, w_cmd = self.control(xyt_err)

                # Command robot
                self._set_velocity(v_cmd, w_cmd)

            # Spin
            rate.sleep()

    def main(self):
        # Subscribers
        rospy.Subscriber(
            "/state_estimator/pose_filtered",
            PoseStamped,
            self._pose_update_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            "/goto_controller/goal", Pose, self._goal_update_callback, queue_size=1
        )
        rospy.Subscriber(
            "/goto_controller/yaw_tracking",
            Bool,
            self._yaw_toggle_callback,
            queue_size=1,
        )

        # Run controller
        self._run_control_loop()


if __name__ == "__main__":
    node = GotoVelocityController(CONTROL_HZ)
    node.main()
