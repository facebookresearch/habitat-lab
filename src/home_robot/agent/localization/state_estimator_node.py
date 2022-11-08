import logging
import time
import threading

import numpy as np
import sophus as sp
from scipy.spatial.transform import Rotation as R
import rospy
from geometry_msgs.msg import (
    Pose,
    PoseStamped,
    PoseWithCovarianceStamped,
)
from nav_msgs.msg import Odometry


log = logging.getLogger(__name__)

SLAM_CUTOFF_HZ = 0.2


def pose_ros2sophus(pose):
    r_mat = R.from_quat(
        (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
    ).as_matrix()
    t_vec = np.array([pose.position.x, pose.position.y, pose.position.z])
    return sp.SE3(r_mat, t_vec)


def pose_sophus2ros(pose_se3):
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


class NavStateEstimator:
    def __init__(self):
        self._slam_pose = None
        self._odom = None

        self._filtered_pose = sp.SE3()
        self._slam_pose_sp = sp.SE3()
        self._t_odom_prev = time.time()
        self._pose_odom_prev = sp.SE3()

        self._lock = threading.Lock()

        # Publishers
        rospy.init_node("state_estimator")
        self._estimator_pub = rospy.Publisher(
            "/state_estimator/pose_filtered", PoseStamped, queue_size=1
        )

    def _publish_filtered_state(self, timestamp):
        pose_out = PoseStamped()
        pose_out.header.stamp = timestamp
        pose_out.pose = pose_sophus2ros(self._filtered_pose)
        self._estimator_pub.publish(pose_out)

    def _odom_callback(self, pose):
        t_curr = time.time()
        ros_time = rospy.Time.now()
        with self._lock:
            self._odom = pose

        # Compute injected signals into filtered pose
        pose_odom = pose_ros2sophus(pose.pose.pose)
        pose_diff_odom = self._pose_odom_prev.inverse() * pose_odom
        pose_diff_slam = self._filtered_pose.inverse() * self._slam_pose_sp

        # Update filtered pose
        w = cutoff_angle(t_curr - self._t_odom_prev, SLAM_CUTOFF_HZ)
        coeff = 1 / (w + 1)

        pose_diff_log = (
            coeff * pose_diff_odom.log() + (1 - coeff) * pose_diff_slam.log()
        )
        self._filtered_pose = self._filtered_pose * sp.SE3.exp(pose_diff_log)
        self._publish_filtered_state(ros_time)

        # Update variables
        self._pose_odom_prev = pose_odom
        self._t_odom_prev = t_curr

    def _slam_pose_callback(self, pose):
        with self._lock:
            self._slam_pose = pose

        # Update slam pose for filtering
        self._slam_pose_sp = pose_ros2sophus(pose.pose.pose)

    def run(self):
        # This comes from hector_slam. It's a transform from src_frame = 'base_link', target_frame = 'map'
        rospy.Subscriber(
            "/poseupdate",
            PoseWithCovarianceStamped,
            self._slam_pose_callback,
            queue_size=1,
        )
        # This comes from wheel odometry.
        rospy.Subscriber("/odom", Odometry, self._odom_callback, queue_size=1)

        log.info("State Estimator launched.")
        rospy.spin()


if __name__ == "__main__":
    node = NavStateEstimator()
    node.run()
