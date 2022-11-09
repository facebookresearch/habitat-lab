import logging
from typing import List, Optional
import threading

import numpy as np
import rospy
from std_srvs.srv import Trigger, TriggerResponse
from geometry_msgs.msg import Twist, Pose, PoseStamped
from nav_msgs.msg import Odometry

from home_robot.agent.control.velocity_controllers import DDVelocityControlNoplan
from home_robot.utils.geometry import xyt_global_to_base, sophus2xyt, xyt2sophus
from home_robot.utils.geometry.ros import pose_ros2sophus


log = logging.getLogger(__name__)

CONTROL_HZ = 20


class GotoVelocityController:
    """
    Self-contained controller module for moving a diff drive robot to a target goal.
    Target goal is update-able at any given instant.
    """

    def __init__(
        self,
        hz: float,
        odom_only_feedback: bool = True,
    ):
        self.hz = hz
        self.odom_only = odom_only_feedback

        # Control module
        self.control = DDVelocityControlNoplan(hz)

        # Publishers
        rospy.init_node("goto_controller")
        self.vel_command_pub = rospy.Publisher("stretch/cmd_vel", Twist, queue_size=1)

        # Initialize
        self.xyt_loc = np.zeros(3)
        self.xyt_loc_odom = np.zeros(3)
        self.xyt_goal: Optional[np.ndarray] = None

        self.active = False
        self.track_yaw = True

    def _pose_update_callback(self, msg: PoseStamped):
        pose_sp = pose_ros2sophus(msg.pose)
        self.xyt_loc = sophus2xyt(pose_sp)

    def _odom_update_callback(self, msg: Odometry):
        pose_sp = pose_ros2sophus(msg.pose.pose)
        self.xyt_loc_odom = sophus2xyt(pose_sp)

    def _goal_update_callback(self, msg: Pose):
        pose_sp = pose_ros2sophus(msg)

        if self.odom_only:
            # Project absolute goal from current odometry reading
            pose_delta = xyt2sophus(self.xyt_loc).inverse() * pose_sp
            pose_goal = xyt2sophus(self.xyt_loc_odom) * pose_delta
        else:
            # Assign absolute goal directly
            pose_goal = pose_sp

        self.xyt_goal = sophus2xyt(pose_goal)

    def _yaw_tracking_service(self, request):
        self.track_yaw = not self.track_yaw
        status_str = "ON" if self.track_yaw else "OFF"
        return TriggerResponse(
            success=True,
            message=f"Yaw tracking is now {status_str}",
        )

    def _toggle_on_service(self, request):
        self.active = not self.active
        status_str = "RUNNING" if self.active else "STOPPED"
        return TriggerResponse(
            success=True,
            message=f"Goto controller is now {status_str}",
        )

    def _compute_error_pose(self):
        """
        Updates error based on robot localization
        """
        xyt_loc = self.xyt_loc_odom if self.odom_only else self.xyt_loc
        xyt_err = xyt_global_to_base(self.xyt_goal, self.xyt_loc)
        if not self.track_yaw:
            xyt_err[2] = 0.0

        return xyt_err

    def _set_velocity(self, v_m, w_r):
        cmd = Twist()
        cmd.linear.x = v_m
        cmd.angular.z = w_r
        self.vel_command_pub.publish(cmd)

    def _run_control_loop(self):
        rate = rospy.Rate(self.hz)

        while True:
            if self.active and self.xyt_goal is not None:
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
            "state_estimator/pose_filtered",
            PoseStamped,
            self._pose_update_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            "odom",
            Odometry,
            self._odom_update_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            "goto_controller/goal", Pose, self._goal_update_callback, queue_size=1
        )

        # Services
        rospy.Service("goto_controller/toggle_on", Trigger, self._toggle_on_service)
        rospy.Service(
            "goto_controller/toggle_yaw_tracking", Trigger, self._yaw_tracking_service
        )

        # Run controller
        log.info("Goto Controller launched.")
        self._run_control_loop()


if __name__ == "__main__":
    GotoVelocityController(CONTROL_HZ).main()
