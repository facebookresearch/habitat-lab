import argparse
import pdb

import rospy
from std_srvs.srv import Trigger, TriggerRequest
from std_srvs.srv import SetBool, SetBoolRequest
from geometry_msgs.msg import PoseStamped, Pose, Twist

from home_robot.utils.geometry import xyt2sophus, sophus2xyt
from home_robot.utils.geometry.ros import pose_sophus2ros, pose_ros2sophus


class LocalHelloRobot:
    """
    ROS interface for robot base control
    Currently only works with a local rosmaster
    """

    def __init__(self):
        rospy.init_node("user")

        self._base_state = None

        # Ros pubsub
        self._goal_pub = rospy.Publisher("goto_controller/goal", Pose, queue_size=1)
        self._velocity_pub = rospy.Publisher("stretch/cmd_vel", Twist, queue_size=1)

        self._state_sub = rospy.Subscriber(
            "state_estimator/pose_filtered",
            PoseStamped,
            self._state_callback,
            queue_size=1,
        )

        self._nav_mode_service = rospy.ServiceProxy(
            "/switch_to_navigation_mode", Trigger
        )
        self._pos_mode_service = rospy.ServiceProxy("/switch_to_position_mode", Trigger)
        self._goto_on_service = rospy.ServiceProxy("goto_controller/enable", Trigger)
        self._goto_off_service = rospy.ServiceProxy("goto_controller/disable", Trigger)
        self._set_yaw_service = rospy.ServiceProxy(
            "goto_controller/toggle_yaw_tracking", Trigger
        )

    def set_nav_mode(self):
        """
        Switches to navigation mode.
        Robot always tries to move to goal in nav mode.
        """
        result = self._nav_mode_service(TriggerRequest())
        print(result.message)
        result = self._goto_on_service(TriggerRequest())
        print(result.message)

    def set_pos_mode(self):
        """
        Switches to position mode.
        """
        result = self._pos_mode_service(TriggerRequest())
        print(result.message)
        result = self._goto_off_service(TriggerRequest())
        print(result.message)

    def set_yaw_tracking(self, value: bool = True):
        """
        Turns yaw tracking on/off.
        Robot only tries to reach the xy position of goal if off.
        """
        result = self._set_yaw_service(SetBoolRequest(data=value))
        print(result.message)
        return result.success

    def get_base_state(self):
        """
        Returns base location in the form of [x, y, rz].
        """
        return self._base_state

    def set_goal(self, xyt):
        """
        Sets the goal for the goto controller.
        """
        msg = pose_sophus2ros(xyt2sophus(xyt))
        self._goal_pub.publish(msg)

    def set_velocity(self, v, w):
        """
        Directly sets the linear and angular velocity of robot base.
        Command gets overwritten immediately if goto controller is on.
        """
        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w
        self._velocity_pub.publish(msg)

    # Subscriber callbacks
    def _state_callback(self, msg: PoseStamped):
        self._base_state = sophus2xyt(pose_ros2sophus(msg.pose))


if __name__ == "__main__":
    # Launches an interactive terminal if file is directly run
    robot = LocalHelloRobot()

    import code

    code.interact(local=locals())
