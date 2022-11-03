import time
import math
import threading
import rospy
from std_srvs.srv import Trigger, TriggerRequest
from sensor_msgs.msg import JointState
from control_msgs.msg import FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
from geometry_msgs.msg import PoseStamped, Pose2D, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import hello_helpers.hello_misc as hm
from tf.transformations import euler_from_quaternion
import collections
import numpy as np


class MoveNode(hm.HelloNode):
    def __init__(self):
        hm.HelloNode.__init__(self)
        self.rate = 10.0
        self._joint_state = None
        self._command = None
        self._slam_pose = None
        self._odom = None
        self._linear_movement = collections.deque(maxlen=10)
        self._angular_movement = collections.deque(maxlen=10)
        self._scan_matched_pose = None
        self._lock = threading.Lock()

    def _joint_states_callback(self, joint_state):
        with self._lock:
            self._joint_state = joint_state

    def _slam_pose_callback(self, pose):
        with self._lock:
            self._slam_pose = pose

    def _scan_matched_pose_callback(self, pose):
        with self._lock:
            self._scan_matched_pose = (pose.x, pose.y, pose.theta)

    def _odom_callback(self, pose):
        with self._lock:
            self._odom = pose
            self._linear_movement.append(
                max(abs(pose.twist.twist.linear.x), abs(pose.twist.twist.linear.y))
            )
            self._angular_movement.append(abs(pose.twist.twist.angular.z))

    def is_moving(self):
        with self._lock:
            lm, am = self._linear_movement, self._angular_movement
            max_linear = max(lm)
            max_angular = max(am)
        return max_linear >= 0.05 or max_angular >= 0.05

    def get_slam_pose(self):
        with self._lock:
            pose = self._slam_pose
        if pose is not None:
            quat = np.array(
                [
                    pose.pose.pose.orientation.x,
                    pose.pose.pose.orientation.y,
                    pose.pose.pose.orientation.z,
                    pose.pose.pose.orientation.w,
                ]
            )
            euler = euler_from_quaternion(quat)
            return (pose.pose.pose.position.x, pose.pose.pose.position.y, euler[2])
        else:
            return (0.0, 0.0, 0.0)

    def get_scan_matched_pose(self):
        with self._lock:
            _pose = self._scan_matched_pose
        return _pose

    def get_odom(self):
        with self._lock:
            odom = self._odom
        pose = odom.pose
        quat = np.array(
            [
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w,
            ]
        )
        euler = euler_from_quaternion(quat)
        return (pose.pose.position.x, pose.pose.position.y, euler[2])

    def get_joint_state(self, name=None):
        with self._lock:
            joint_state = self._joint_state
        if joint_state is None:
            return joint_state
        if name is not None:
            joint_index = joint_state.name.index("joint_" + name)
            joint_value = joint_state.position[joint_index]
            return joint_value
        else:
            return joint_state

    def send_command(self, joint_name, increment):
        with self._lock:
            self._command = [joint_name, increment]

    def _send_command(self, joint_name, increment):
        point = JointTrajectoryPoint()
        point.time_from_start = rospy.Duration(0.0)
        point.positions = [increment]

        trajectory_goal = FollowJointTrajectoryGoal()
        trajectory_goal.goal_time_tolerance = rospy.Time(1.0)
        trajectory_goal.trajectory.joint_names = [joint_name]
        trajectory_goal.trajectory.points = [point]
        trajectory_goal.trajectory.header.stamp = rospy.Time.now()

        self.trajectory_client.send_goal(trajectory_goal)

        rospy.loginfo(
            "joint_name = {0}, trajectory_goal = {1}".format(joint_name, trajectory_goal)
        )
        rospy.loginfo("Done sending pose.")

    def stop(self):
        rospy.wait_for_service("stop_the_robot")
        s = rospy.ServiceProxy("stop_the_robot", Trigger)
        s_request = TriggerRequest()
        result = s(s_request)
        return result

    def background_loop(self):

        rospy.Subscriber(
            "/stretch/joint_states", JointState, self._joint_states_callback, queue_size=1
        )
        # This comes from hector_slam. It's a transform from src_frame = 'base_link', target_frame = 'map'
        rospy.Subscriber(
            "/poseupdate", PoseWithCovarianceStamped, self._slam_pose_callback, queue_size=1
        )
        # this comes from lidar matching, i.e. no slam/global-optimization
        rospy.Subscriber("/pose2D", Pose2D, self._scan_matched_pose_callback, queue_size=1)
        # This comes from wheel odometry.
        rospy.Subscriber("/odom", Odometry, self._odom_callback, queue_size=1)

        rate = rospy.Rate(self.rate)

        while not rospy.is_shutdown():
            with self._lock:
                command = self._command
                self._command = None
            if command is not None:
                joint_name, increment = command
                self._send_command(joint_name, increment)
            rate.sleep()

    def start(self):
        hm.HelloNode.main(
            self, "fairo_hello_proxy", "fairo_hello_proxy", wait_for_first_pointcloud=False
        )
        self._thread = threading.Thread(target=self.background_loop, daemon=True)
        self._thread.start()
        # self.send_command('translate_mobile_base', 0.1)
        # self.send_command('rotate_mobile_base', math.radians(90))
        # while not rospy.is_shutdown():
        #     time.sleep(1)
        #     # self.send_command('rotate_mobile_base', math.radians(90))
        #     #     self.send_command('rotate_mobile_base', math.radians(6))
        #     print("")
        #     print('slam_pose', self.get_slam_pose())
        #     print('odom', self.get_odom())
        #     print('scan_matched_pose', self.get_scan_matched_pose())
        #     print("")
        #     # ROTATE LEFT +ve / RIGHT -ve in radians
        #     self.send_command('rotate_mobile_base', math.radians(6))
        #     # print(self.get_joint_state('head_pan'))
        #     # joint_state = self.get_joint_state()
        #     # if joint_state is not None:
        #     #     print(joint_state)
        #     # slam_pose = self.get_slam_pose()
        #     # if slam_pose is not None:
        #     #     # print(slam_pose)
        #     #     x = slam_pose.pose.position.x
        #     #     y = slam_pose.pose.position.y
        #     #     theta = slam_pose.pose.orientation.z
        #     #     print(x, y, theta)

        #     # FORWARD/BACKWARD in metres
        # self.send_command('translate_mobile_base', 0.05)


if __name__ == "__main__":
    node = MoveNode()
    node.start()
    # node.send_command('rotate_mobile_base', math.radians(6))
