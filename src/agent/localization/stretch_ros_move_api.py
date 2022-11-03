# NOTE: Integrate the state estimator into localization!

import time
import math
import threading
import rospy
from std_srvs.srv import Trigger, TriggerRequest
from sensor_msgs.msg import JointState
from control_msgs.msg import FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
from geometry_msgs.msg import Pose, PoseStamped, Pose2D, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
import hello_helpers.hello_misc as hm
from tf.transformations import euler_from_quaternion
import collections
import numpy as np
import sophus as sp
from scipy.spatial.transform import Rotation as R

import rospy
from geometry_msgs.msg import Twist


SLAM_CUTOFF_HZ = 0.2


def pose_ros2sp(pose):
    r_mat = R.from_quat(
        (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
    ).as_matrix()
    t_vec = np.array([pose.position.x, pose.position.y, pose.position.z])
    return sp.SE3(r_mat, t_vec)


def pose_sp2ros(pose_se3):
    quat = R.from_matrix(pose_se3.so3().matrix()).as_quat()

    pose = Pose()
    pose.position.x = pose_se3.translation()[0]
    pose.position.y = pose_se3.translation()[1]
    pose.position.z = pose_se3.translation()[2]
    pose.orientation.x = quat[0]
    pose.orientation.y = quat[1]
    pose.orientation.z = quat[2]
    pose.orientation.w = quat[3]

    return pose


def cutoff_angle(duration, cutoff_freq):
    return 2 * np.pi * duration * cutoff_freq


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

        self._filtered_pose = sp.SE3()
        self._slam_pose_sp = sp.SE3()
        self._t_odom_prev = time.time()
        self._pose_odom_prev = sp.SE3()

        self._nav_mode = rospy.ServiceProxy("/switch_to_navigation_mode", Trigger)
        s_request = TriggerRequest()
        self._nav_mode(s_request)

        self._vel_command_pub = rospy.Publisher("/stretch/cmd_vel", Twist, queue_size=1)
        self._estimator_pub = rospy.Publisher(
            "/state_estimator/pose_filtered", PoseStamped, queue_size=1
        )

    def _joint_states_callback(self, joint_state):
        with self._lock:
            self._joint_state = joint_state

    def _slam_pose_callback(self, pose):
        t_curr = time.time()
        ros_time = rospy.Time.now()
        with self._lock:
            self._slam_pose = pose

        # Update slam pose for filtering
        self._slam_pose_sp = pose_ros2sp(pose.pose.pose)

    def _scan_matched_pose_callback(self, pose):
        with self._lock:
            self._scan_matched_pose = (pose.x, pose.y, pose.theta)

    def _odom_callback(self, pose):
        t_curr = time.time()
        ros_time = rospy.Time.now()
        with self._lock:
            self._odom = pose
            self._linear_movement.append(
                max(abs(pose.twist.twist.linear.x), abs(pose.twist.twist.linear.y))
            )
            self._angular_movement.append(abs(pose.twist.twist.angular.z))

        # Compute injected signals into filtered pose
        pose_odom = pose_ros2sp(pose.pose.pose)
        pose_diff_odom = self._pose_odom_prev.inverse() * pose_odom
        pose_diff_slam = self._filtered_pose.inverse() * self._slam_pose_sp

        # Update filtered pose
        w = cutoff_angle(t_curr - self._t_odom_prev, SLAM_CUTOFF_HZ)
        coeff = 1 / (w + 1)

        pose_diff_log = coeff * pose_diff_odom.log() + (1 - coeff) * pose_diff_slam.log()
        self._filtered_pose = self._filtered_pose * sp.SE3.exp(pose_diff_log)
        self._publish_filtered_state(ros_time)

        # Update variables
        self._pose_odom_prev = pose_odom
        self._t_odom_prev = t_curr

    def _publish_filtered_state(self, timestamp):
        pose_out = PoseStamped()
        pose_out.header.stamp = timestamp
        pose_out.pose = pose_sp2ros(self._filtered_pose)
        self._estimator_pub.publish(pose_out)

    def set_velocity(self, v_m, w_r):
        cmd = Twist()
        cmd.linear.x = v_m
        cmd.angular.z = w_r
        self._vel_command_pub.publish(cmd)

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

    def get_estimator_pose(self):
        with self._lock:
            pose = self._filtered_pose.copy()

        t_vec = pose.translation()
        r_vec = pose.so3().log()
        return (t_vec[0], t_vec[1], r_vec[2])

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
