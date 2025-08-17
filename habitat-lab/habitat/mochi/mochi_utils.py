# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
import numpy as np

# kitchen counter
# HABITAT_TO_MOCHI_POS_OFFSET = -mn.Vector3(-3.7, 0.956, -3.0)

# living coffee table
# HABITAT_TO_MOCHI_POS_OFFSET = -mn.Vector3(-8.05, 0.465, -3.94)

# dining table
# HABITAT_TO_MOCHI_POS_OFFSET = -mn.Vector3(-3.55631, 0.76268, -7.34351)

# modern kitchen
# HABITAT_TO_MOCHI_POS_OFFSET = -mn.Vector3(0.0, 1.0, 0.0)

# at origin, good for debugging in polyscope
HABITAT_TO_MOCHI_POS_OFFSET = -mn.Vector3(0.0, 0.0, 0.0)
# raised up 1 meter to make VR interaction easier?
# HABITAT_TO_MOCHI_POS_OFFSET = -mn.Vector3(0.0, 0.0, 0.0)

def habitat_to_mochi_position(src):
    if isinstance(src, mn.Vector3):
        return src + HABITAT_TO_MOCHI_POS_OFFSET
    else:
        return [
            src[0] + HABITAT_TO_MOCHI_POS_OFFSET[0],
            src[1] + HABITAT_TO_MOCHI_POS_OFFSET[1],
            src[2] + HABITAT_TO_MOCHI_POS_OFFSET[2],
        ]


def mochi_to_habitat_position(src):
    if isinstance(src, mn.Vector3):
        return src - HABITAT_TO_MOCHI_POS_OFFSET
    else:
        return [
            src[0] - HABITAT_TO_MOCHI_POS_OFFSET[0],
            src[1] - HABITAT_TO_MOCHI_POS_OFFSET[1],
            src[2] - HABITAT_TO_MOCHI_POS_OFFSET[2],
        ]


def rotvec_to_quat_wxyz(rotvec):
    theta = np.linalg.norm(rotvec)
    if theta < 1e-8:
        # No rotation, return identity quaternion
        return np.array([1.0, 0.0, 0.0, 0.0])  # [x, y, z, w]

    axis = rotvec / theta
    half_theta = theta / 2.0
    sin_half_theta = np.sin(half_theta)
    cos_half_theta = np.cos(half_theta)

    q_xyz = axis * sin_half_theta
    q_w = cos_half_theta
    return np.concatenate([[q_w], q_xyz])


def rotvec_to_magnum_quat(rotvec):
    q_wxyz = rotvec_to_quat_wxyz(rotvec)  # [w, x, y, z]
    x, y, z = q_wxyz[1:4]
    w = q_wxyz[0]
    return mn.Quaternion(mn.Vector3(x, y, z), w)

def quat_to_rotvec(q) -> np.ndarray:

    if isinstance(q, mn.Quaternion):
        q = q.normalized()
        # Always use positive w for consistency (same as C++ code)
        v = np.array(list(q.vector))
        w = q.scalar
    elif isinstance(q, (list, np.array)):
        # assume wxyz
        v = np.array(q[1:])
        w = q[0]

    if w < 0:
        v = -v
        w = -w

    mag = np.linalg.norm(v)
    if w < mag:
        angle = 2.0 * np.arccos(w)
    else:
        angle = 2.0 * np.arcsin(mag)

    if mag > 1e-9:
        axis = v / mag
    else:
        axis = np.array([1.0, 0.0, 0.0])  # fallback

    rotvec = axis * angle

    return rotvec


def rotvec_to_quat_wxyz(rotvec):
    theta = np.linalg.norm(rotvec)
    if theta < 1e-8:
        # No rotation, return identity quaternion
        return np.array([1.0, 0.0, 0.0, 0.0])  # [x, y, z, w]

    axis = rotvec / theta
    half_theta = theta / 2.0
    sin_half_theta = np.sin(half_theta)
    cos_half_theta = np.cos(half_theta)

    q_xyz = axis * sin_half_theta
    q_w = cos_half_theta
    return np.concatenate([[q_w], q_xyz])


def magnum_quat_to_list_wxyz(rot_quat):
    return [rot_quat.scalar, *list(rot_quat.vector)]

def list_wxyz_to_magnum_quat(q_wxyz):
    return mn.Quaternion(mn.Vector3(q_wxyz[1], q_wxyz[2], q_wxyz[3]), q_wxyz[0])    


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return [w, x, y, z]


def quat_transpose(q):
    w, x, y, z = q
    return [w, -x, -y, -z]


def quat_rotate(q, v):
    q_conj = quat_transpose(q)
    v_quat = [0, *v]
    return quat_multiply(quat_multiply(q, v_quat), q_conj)[1:]


def quat_to_rotmat(rotation_wxyz):
    """Convert quaternion [wxyz] to 3x3 rotation matrix."""
    w, x, y, z = rotation_wxyz
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(xx + zz),       2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx),   1 - 2*(xx + yy)]
    ])

def world_to_local(point_world, pos, quat):
    """Transform point from world to local coordinates.

    Args:
        point_world: (3,) array
        pos: (3,) array, world position of local origin
        quat: (4,) array [x, y, z, w], local→world orientation

    Returns:
        point_local: (3,) array
    """
    R = quat_to_rotmat(quat)
    return R.T @ (point_world - pos)

def local_to_world(point_local, pos, quat):
    """Transform point from local to world coordinates.

    Args:
        point_local: (3,) array
        pos: (3,) array, world position of local origin
        quat: (4,) array [x, y, z, w], local→world orientation

    Returns:
        point_world: (3,) array
    """
    R = quat_to_rotmat(quat)
    return R @ point_local + pos