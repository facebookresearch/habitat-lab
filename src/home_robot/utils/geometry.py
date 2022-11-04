import numpy as np
import sophus as sp
from scipy.spatial.transform import Rotation


def xyt_global_to_base(XYT, current_pose):
    """
    Transforms the point cloud into geocentric frame to account for
    camera position
    Input:
        XYZ                     : ...x3
        current_pose            : base position (x, y, theta (radians))
    Output:
        XYZ : ...x3
    """
    XYT = np.asarray(XYT, dtype=np.float32)
    new_T = XYT[2] - current_pose[2]
    R = Rotation.from_euler("Z", current_pose[2]).as_matrix()
    XYT[0] = XYT[0] - current_pose[0]
    XYT[1] = XYT[1] - current_pose[1]
    out_XYT = np.matmul(XYT.reshape(-1, 3), R).reshape((-1, 3))
    out_XYT = out_XYT.ravel()
    return [out_XYT[0], out_XYT[1], new_T]


def xyt_base_to_global(out_XYT, current_pose):
    """
    Transforms the point cloud from base frame into geocentric frame
    Input:
        XYZ                     : ...x3
        current_pose            : base position (x, y, theta (radians))
    Output:
        XYZ : ...x3
    """
    R = Rotation.from_euler("Z", current_pose[2]).as_matrix()
    Rinv = np.linalg.inv(R)

    XYT = np.matmul(R, out_XYT)

    XYT[0] = XYT[0] + current_pose[0]
    XYT[1] = XYT[1] + current_pose[1]

    XYT[2] = out_XYT[2] + current_pose[2]

    XYT = np.asarray(XYT)

    return XYT


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
