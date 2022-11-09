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


def xyt2sophus(xyt: np.ndarray) -> sp.SE3:
    """
    Converts SE2 coordinates (x, y, rz) to an sophus SE3 pose object.
    """
    x = np.array([xyt[0], xyt[1], 0.0])
    r_mat = sp.SO3.exp([0.0, 0.0, xyt[2]]).matrix()
    return sp.SE3(r_mat, x)


def sophus2xyt(se3: sp.SE3) -> np.ndarray:
    """
    Converts an sophus SE3 pose object to SE2 coordinates (x, y, rz).
    """
    x_vec = se3.translation()
    r_vec = se3.so3().log()
    return np.array([x_vec[0], x_vec[1], r_vec[2]])
