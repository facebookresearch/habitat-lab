#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import quaternion

from habitat.core.simulator import AgentState
from habitat.tasks.utils import quaternion_from_coeff, quaternion_rotate_vector

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


def agent_state_target2ref(
    ref_agent_state: AgentState, target_agent_state: AgentState
) -> AgentState:
    r"""Computes the target agent_state's position and rotation representation
    with respect to the coordinate system defined by reference agent's position and rotation.

    :param ref_agent_state: reference agent_state,
        whose global position and rotation attributes define a local coordinate system.
    :param target_agent_state: target agent_state,
        whose global position and rotation attributes need to be transformed to
        the local coordinate system defined by ref_agent_state.
    """

    target_in_ref_coordinate = AgentState(
        position=np.zeros(3), rotation=np.quaternion(1, 0, 0, 0)
    )

    if isinstance(ref_agent_state.rotation, (List, np.ndarray)):
        ref_agent_state.rotation = quaternion_from_coeff(
            ref_agent_state.rotation
        )
    if isinstance(target_agent_state.rotation, (List, np.ndarray)):
        target_agent_state.rotation = quaternion_from_coeff(
            target_agent_state.rotation
        )

    target_in_ref_coordinate.rotation = (
        ref_agent_state.rotation.inverse() * target_agent_state.rotation
    )

    target_in_ref_coordinate.position = quaternion_rotate_vector(
        ref_agent_state.rotation.inverse(),
        target_agent_state.position - ref_agent_state.position,
    )

    return target_in_ref_coordinate
