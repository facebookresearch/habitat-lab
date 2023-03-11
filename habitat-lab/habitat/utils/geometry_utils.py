#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import List, Tuple, Union

import numpy as np
import quaternion

EPSILON = 1e-8


def angle_between_quaternions(
    q1: quaternion.quaternion, q2: quaternion.quaternion
) -> float:
    r"""Returns the angle (in radians) between two quaternions. This angle will
    always be positive.
    """
    q1_inv = np.conjugate(q1)
    dq = quaternion.as_float_array(q1_inv * q2)

    return 2 * np.arctan2(np.linalg.norm(dq[1:]), np.abs(dq[0]))


def quaternion_from_two_vectors(
    v0: np.ndarray, v1: np.ndarray
) -> quaternion.quaternion:
    r"""Computes the quaternion representation of v1 using v0 as the origin."""
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
        return quaternion.quaternion(w, *axis)

    axis = np.cross(v0, v1)
    s = np.sqrt((1 + c) * 2)
    return quaternion.quaternion(s * 0.5, *(axis / s))


def quaternion_to_list(q: quaternion.quaternion):
    return q.imag.tolist() + [q.real]


def quaternion_from_coeff(coeffs: List[float]) -> quaternion.quaternion:
    r"""Creates a quaternions from coeffs in [x, y, z, w] format"""
    quat = quaternion.quaternion(0, 0, 0, 0)
    quat.real = coeffs[3]
    quat.imag = coeffs[0:3]
    return quat


def quaternion_rotate_vector(
    quat: quaternion.quaternion, v: np.ndarray
) -> np.ndarray:
    r"""Rotates a vector by a quaternion
    Args:
        quaternion: The quaternion to rotate by
        v: The vector to rotate
    Returns:
        np.ndarray: The rotated vector
    """
    vq = quaternion.quaternion(0, 0, 0, 0)
    vq.imag = v
    return (quat * vq * quat.inverse()).imag


def agent_state_target2ref(
    ref_agent_state: Union[List, Tuple], target_agent_state: Union[List, Tuple]
) -> Tuple[quaternion.quaternion, np.ndarray]:
    r"""Computes the target agent_state's rotation and position representation
    with respect to the coordinate system defined by reference agent's rotation and position.
    All rotations must be in [x, y, z, w] format.

    :param ref_agent_state: reference agent_state in the format of [rotation, position].
         The rotation and position are from a common/global coordinate systems.
         They define a local coordinate system.
    :param target_agent_state: target agent_state in the format of [rotation, position].
        The rotation and position are from a common/global coordinate systems.
        and need to be transformed to the local coordinate system defined by ref_agent_state.
    """

    assert (
        len(ref_agent_state[1]) == 3
    ), "Only support Cartesian format currently."
    assert (
        len(target_agent_state[1]) == 3
    ), "Only support Cartesian format currently."

    ref_rotation, ref_position = ref_agent_state
    target_rotation, target_position = target_agent_state

    # convert to all rotation representations to np.quaternion
    if not isinstance(ref_rotation, quaternion.quaternion):
        ref_rotation = quaternion_from_coeff(ref_rotation)
    ref_rotation = ref_rotation.normalized()

    if not isinstance(target_rotation, quaternion.quaternion):
        target_rotation = quaternion_from_coeff(target_rotation)
    target_rotation = target_rotation.normalized()

    rotation_in_ref_coordinate = ref_rotation.inverse() * target_rotation

    position_in_ref_coordinate = quaternion_rotate_vector(
        ref_rotation.inverse(), target_position - ref_position
    )

    return (rotation_in_ref_coordinate, position_in_ref_coordinate)


def random_triangle_point(
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray
) -> np.ndarray:
    """
    Sample a random point from a triangle given its vertices.
    """

    # reference: https://mathworld.wolfram.com/TrianglePointPicking.html
    coef1 = random.random()
    coef2 = random.random()
    if coef1 + coef2 >= 1:
        # transform "outside" points back inside
        coef1 = 1 - coef1
        coef2 = 1 - coef2
    return v0 + coef1 * (v1 - v0) + coef2 * (v2 - v0)


def is_point_in_triangle(
    p: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray
) -> bool:
    """
    Return True if the point, p, is in the triangle defined by vertices v0,v1,v2.
    Algorithm: https://gdbooks.gitbooks.io/3dcollisions/content/Chapter4/point_in_triangle.html
    """
    # 1. move the triangle such that point is the origin
    a = v0 - p
    b = v1 - p
    c = v2 - p

    # 2. check that the origin is planar
    tri_norm = np.cross(c - a, b - a)
    # NOTE: small epsilon error allowed here empirically
    if abs(np.dot(a, tri_norm)) > 1e-7:
        return False

    # 3. create 3 triangles with origin + pairs of vertices and compute the normals
    u = np.cross(b, c)
    v = np.cross(c, a)
    w = np.cross(a, b)

    # 4. check that all new triangle normals are aligned
    if np.dot(u, v) < 0.0:
        return False
    if np.dot(u, w) < 0.0:
        return False
    if np.dot(v, w) < 0.0:
        return False
    return True
