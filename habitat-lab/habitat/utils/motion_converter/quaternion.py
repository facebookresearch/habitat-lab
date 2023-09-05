# Copyright (c) Facebook, Inc. and its affiliates.

import math
import numpy as np

from . import constants,utils,conversions,calc as math_ops

from scipy.spatial.transform import Rotation


def Q_op(Q, op, xyzw_in=True):
    """
    Perform operations on quaternion. The operations currently supported are
    "change_order", "normalize" and "halfspace".

    `change_order` changes order of quaternion to xyzw if it's in wxyz and
    vice-versa
    `normalize` divides the quaternion by its norm
    `half-space` negates the quaternion if w < 0

    Args:
        Q: Numpy array of shape (..., 4)
        op: String; The operation to be performed on the quaternion. `op` can
            take values "change_order", "normalize" and "halfspace"
        xyzw_in: Set to True if input order is "xyzw". Otherwise, the order
            "wxyz" is assumed.
    """

    def q2q(q):
        result = q.copy()
        if "normalize" in op:
            norm = np.linalg.norm(result)
            if norm < constants.EPSILON:
                raise Exception("Invalid input with zero length")
            result /= norm
        if "halfspace" in op:
            w_idx = 3 if xyzw_in else 0
            if result[w_idx] < 0.0:
                result *= -1.0
        if "change_order" in op:
            result = result[[3, 0, 1, 2]] if xyzw_in else result[[1, 2, 3, 0]]
        return result

    return utils._apply_fn_agnostic_to_vec_mat(Q, q2q)


def Q_diff(Q1, Q2):
    raise NotImplementedError


def Q_mult(Q1, Q2):
    """
    Multiply two quaternions.
    """
    R1 = Rotation.from_quat(Q1)
    R2 = Rotation.from_quat(Q2)
    return (R1 * R2).as_quat()


def Q_closest(Q1, Q2, axis):
    """
    This computes optimal-in-place orientation given a target orientation Q1
    and a geodesic curve (Q2, axis). In tutively speaking, the optimal-in-place
    orientation is the closest orientation to Q1 when we are able to rotate Q2
    along the given axis. We assume Q is given in the order of xyzw.
    """
    ws, vs = Q1[3], Q1[0:3]
    w0, v0 = Q2[3], Q2[0:3]
    u = math_ops.normalize(axis)

    a = ws * w0 + np.dot(vs, v0)
    b = -ws * np.dot(u, v0) + w0 * np.dot(vs, u) + np.dot(vs, np.cross(u, v0))
    alpha = math.atan2(a, b)

    theta1 = -2 * alpha + math.pi
    theta2 = -2 * alpha - math.pi
    G1 = conversions.A2Q(theta1 * u)
    G2 = conversions.A2Q(theta2 * u)

    if np.dot(Q1, G1) > np.dot(Q1, G2):
        theta = theta1
        Qnearest = Q_mult(G1, Q2)
    else:
        theta = theta2
        Qnearest = Q_mult(G1, Q2)

    return Qnearest, theta
