#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import quaternion

EPSILON = 1e-8


def angle_between_quaternions(q1: np.quaternion, q2: np.quaternion) -> float:
    r"""Returns the angle (in radians) between two quaternions. This angle will
    always be positive.
    """
    q1_inv = np.conjugate(q1)
    dq = quaternion.as_float_array(q1_inv * q2)

    return 2 * np.arctan2(np.linalg.norm(dq[1:]), np.abs(dq[0]))


def quaternion_from_two_vectors(v0: np.array, v1: np.array) -> np.quaternion:
    r"""Computes the quaternion representation of v1 using v0 as the origin.
    """
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    c = v0.dot(v1)
    # Epsilon prevents issues at poles.
    if c < (-1 + EPSILON):
        c = max(c, -1)
        m = np.stack([v0, v1], 0)
        _, _, vh = np.linalg.svd(m, full_matrices=True)
        axis = vh.T[:, 2]
        w2 = (1 + c) * 0.5
        w = np.sqrt(w2)
        axis = axis * np.sqrt(1 - w2)
        return np.quaternion(w, *axis)

    axis = np.cross(v0, v1)
    s = np.sqrt((1 + c) * 2)
    return np.quaternion(s * 0.5, *(axis / s))


def quaternion_xyzw_to_wxyz(v: np.array):
    return np.quaternion(v[3], *v[0:3])


def quaternion_wxyz_to_xyzw(v: np.array):
    return np.quaternion(*v[1:4], v[0])


def quaternion_to_list(q: np.quaternion):
    return quaternion.as_float_array(
        quaternion_wxyz_to_xyzw(quaternion.as_float_array(q))
    ).tolist()
