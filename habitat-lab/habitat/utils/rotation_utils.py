import math
from typing import List, Tuple, Union

import magnum as mn
import numpy as np
import quaternion


def wrap_heading(heading):
    """Ensures input heading is between -180 an 180; can be float or np.ndarray"""
    return (heading + np.pi) % (2 * np.pi) - np.pi


def convert_conventions(
    mat_or_vec: Union[mn.Matrix4, mn.Vector3, np.ndarray],
    reverse: bool = False,
) -> Union[mn.Matrix4, mn.Vector3]:
    """
    Convert a Matrix4 or Vector3 from Habitat convention (x=right, y=up, z=backwards)
    to standard convention (x=forward, y=left, z=up), or vice versa.

    Args:
        mat_or_vec (Union[mn.Matrix4, mn.Vector3]): The input to convert, either a 4x4
            transformation matrix or a 3D vector in the source coordinate system
        reverse (bool, optional): If True, converts from standard to Habitat convention.
            If False, converts from Habitat to standard convention. Defaults to False.

    Returns:
        Union[mn.Matrix4, mn.Vector3]: The converted matrix or vector in the target
        coordinate system. Returns the same type as the input (Matrix4 or Vector3).
    """
    if not isinstance(mat_or_vec, np.ndarray):
        mat_or_vec_in = np.array(mat_or_vec)
    else:
        mat_or_vec_in = mat_or_vec

    assert mat_or_vec_in.shape in [
        (3,),
        (4, 4),
    ], "Invalid shape for input array"

    perm = np.array(
        [
            [0.0, 0.0, -1.0],  # New x comes from negated old z
            [-1.0, 0.0, 0.0],  # New y comes from negated old x
            [0.0, 1.0, 0.0],  # New z comes from old y
        ]
    )

    if reverse:
        perm = perm.T

    if mat_or_vec_in.shape == (3,):
        return mn.Vector3(perm @ mat_or_vec_in)

    result = np.eye(4)
    result[:3, :] = perm @ mat_or_vec_in[:3, :]
    return mn.Matrix4(result)


def extract_roll_pitch_yaw(matrix):
    """
    Extract roll, pitch, and yaw angles in radians from a 4x4 transformation matrix.

    Args:
        matrix (numpy.ndarray): 4x4 transformation matrix

    Returns:
        tuple: (roll, pitch, yaw) in radians
    """
    # Extract rotation submatrix (top-left 3x3)
    R = np.array(matrix)[:3, :3]

    # Extract pitch (rotation around Y axis)
    pitch = math.atan2(-R[2, 0], math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))

    # Handle gimbal lock case
    if abs(pitch) > np.pi / 2 - 1e-6:
        # Gimbal lock: set roll to 0 and compute yaw
        roll = 0
        yaw = math.atan2(R[1, 2], R[1, 1])
    else:
        # Extract roll (rotation around X axis)
        roll = math.atan2(R[2, 1], R[2, 2])

        # Extract yaw (rotation around Z axis)
        yaw = math.atan2(R[1, 0], R[0, 0])

    return roll, pitch, yaw


def transform_position(pos, direction="sim_to_real"):
    if direction == "sim_to_real":
        return np.array([-pos[2], pos[0], pos[1]])
    elif direction == "real_to_sim":
        return np.array([-pos[1], pos[2], pos[0]])
