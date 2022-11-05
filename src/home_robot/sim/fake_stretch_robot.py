"""
Launches a kinematic simulation that mimics:
- stretch_ros: Publishes odometry information, subscribes to velocity commands
- Hector slam: Publishes slam pose information
"""
import time

import numpy as np
from scipy.spatial.transform import Rotation as R
import rospy
from geometry_msgs.msg import (
    PoseWithCovarianceStamped,
    Twist,
)
from nav_msgs.msg import Odometry

from home_robot.utils.geometry import xyt2sophus
from home_robot.utils.ros_geometry import pose_sophus2ros


SIM_HZ = 240
VEL_CONTROL_HZ = 20


class Env:
    def __init__(self, hz):
        self.pos_state = np.zeros(3)
        self.vel_state = np.zeros(2)
        self.dt = 1.0 / hz

    def get_pose(self):
        return self.pos_state

    def step(self, vel_input=None):
        if vel_input is not None:
            self.vel_state = vel_input

        self.pos_state[0] += self.vel_state[0] * np.cos(self.pos_state[2]) * self.dt
        self.pos_state[1] += self.vel_state[0] * np.sin(self.pos_state[2]) * self.dt
        self.pos_state[2] += self.vel_state[1] * self.dt


class FakeStretch:
    def __init__(self, sim_hz, control_hz):
        self.sim = Env(sim_hz)
        self.control_hz = control_hz
        self._vel_cmd_cache = [0.0, 0.0]

        # Ros stuff
        rospy.init_node("fake_stretch_hw")
        self._hector_slam_pub = rospy.Publisher(
            "/poseupdate",
            PoseWithCovarianceStamped,
            queue_size=1,
        )
        self._odom_pub = rospy.Publisher("/odom", Odometry, queue_size=1)

    def _publish_slam(self, xyt, timestamp):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = timestamp
        msg.pose.pose = pose_sophus2ros(xyt2sophus(xyt))
        self._hector_slam_pub.publish(msg)

    def _publish_odom(self, xyt, timestamp):
        msg = Odometry()
        msg.header.stamp = timestamp
        msg.pose.pose = pose_sophus2ros(xyt2sophus(xyt))

    def _vel_control_callback(self, cmd_vel):
        self._vel_cmd_cache[0] = cmd_vel.linear.x
        self._vel_cmd_cache[1] = cmd_vel.angular.z

    def run(self):
        # Subscribers
        rospy.Subscriber(
            "/stretch/cmd_vel", Twist, self._vel_control_callback, queue_size=1
        )

        # Sim loop
        dt_control = 1 / self.control_hz

        rate = rospy.Rate(self.control_hz)
        t_control_target = time.time()
        while True:
            # Publish states
            ros_time = rospy.Time.now()
            self._publish_odom(self.sim.get_pose(), ros_time)
            self._publish_slam(self.sim.get_pose(), ros_time)

            # Apply control at control_hz
            vel_cmd = None
            if t_control_target <= time.time():
                vel_cmd = self._vel_cmd_cache
                t_control_target += dt_control

            # Step sim
            self.sim.step(vel_cmd)


if __name__ == "__main__":
    sim = FakeStretch(SIM_HZ, VEL_CONTROL_HZ)
    sim.run()
