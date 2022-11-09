import argparse
import pdb

import rospy
from std_srvs.srv import Trigger, TriggerRequest
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

        # Publishers
        self._goal_pub = rospy.Publisher("goto_controller/goal", Pose, queue_size=1)
        self._velocity_pub = rospy.Publisher("stretch/cmd_vel", Twist, queue_size=1)

        # Services
        self._goto_service = rospy.ServiceProxy("goto_controller/toggle_on", Trigger)
        self._yaw_service = rospy.ServiceProxy(
            "goto_controller/toggle_yaw_tracking", Trigger
        )

        # Subscribers
        self._state_sub = rospy.Subscriber(
            "state_estimator/pose_filtered",
            PoseStamped,
            self._state_callback,
            queue_size=1,
        )

    def toggle_controller(self):
        """
        Turns goto controller on/off.
        Robot always tries to move to goal if on.
        """
        result = self._goto_service(TriggerRequest())
        print(result.message)
        return result.success

    def toggle_yaw_tracking(self):
        """
        Turns yaw tracking on/off.
        Robot only tries to reach the xy position of goal if off.
        """
        result = self._yaw_service(TriggerRequest())
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
