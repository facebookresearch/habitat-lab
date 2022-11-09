import logging

import numpy as np
import sophus as sp
from scipy.spatial.transform import Rotation

from geometry_msgs.msg import Pose

log = logging.getLogger(__name__)


def pose_ros2sophus(pose):
    quat = np.array(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    )
    if np.linalg.norm(quat) == 0.0:
        log.warning(
            "Zero-norm quaternion received. Automatically normalizing to [0, 0, 0, 1]..."
        )
        quat[3] = 1.0

    r_mat = Rotation.from_quat(quat).as_matrix()
    t_vec = np.array([pose.position.x, pose.position.y, pose.position.z])

    return sp.SE3(r_mat, t_vec)


def pose_sophus2ros(pose_se3):
    quat = Rotation.from_matrix(pose_se3.so3().matrix()).as_quat()

    pose = Pose()
    pose.position.x = pose_se3.translation()[0]
    pose.position.y = pose_se3.translation()[1]
    pose.position.z = pose_se3.translation()[2]
    pose.orientation.x = quat[0]
    pose.orientation.y = quat[1]
    pose.orientation.z = quat[2]
    pose.orientation.w = quat[3]

    return pose
