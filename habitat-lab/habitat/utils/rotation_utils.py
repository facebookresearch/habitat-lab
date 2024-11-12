import math

import numpy as np
import quaternion
import magnum as mn

def euler_to_quaternion(euler):
    """Convert Euler angles to quaternion."""
    roll, pitch, yaw = euler

    cr, cp, cy = np.cos(roll / 2), np.cos(pitch / 2), np.cos(yaw / 2)
    sr, sp, sy = np.sin(roll / 2), np.sin(pitch / 2), np.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


def euler_to_matrix(euler: tuple) -> np.ndarray:
    """Convert euler angles to 3x3 rotation matrix_in. Euler angles in radians in ZYX order."""
    roll, pitch, yaw = euler

    # Roll (X-axis rotation)
    Rx = np.array(
        [
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll), math.cos(roll)],
        ]
    )

    # Pitch (Y-axis rotation)
    Ry = np.array(
        [
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)],
        ]
    )

    # Yaw (Z-axis rotation)
    Rz = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1],
        ]
    )

    return Rz @ Ry @ Rx


def quaternion_to_euler(quat) -> tuple:
    """Convert quaternion (w,x,y,z) to euler angles in radians (ZYX order)."""
    if isinstance(quat, quaternion.quaternion):
        quat = np.array([quat.w, quat.x, quat.y, quat.z])

    w, x, y, z = quat

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(
            math.pi / 2, sinp
        )  # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return (roll, pitch, yaw)


def quaternion_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (w,x,y,z) to 3x3 rotation matrix_in."""
    if isinstance(quat, quaternion.quaternion):
        quat = np.array([quat.w, quat.x, quat.y, quat.z])
    w, x, y, z = quat

    return np.array(
        [
            [
                1 - 2 * y * y - 2 * z * z,
                2 * x * y - 2 * w * z,
                2 * x * z + 2 * w * y,
            ],
            [
                2 * x * y + 2 * w * z,
                1 - 2 * x * x - 2 * z * z,
                2 * y * z - 2 * w * x,
            ],
            [
                2 * x * z - 2 * w * y,
                2 * y * z + 2 * w * x,
                1 - 2 * x * x - 2 * y * y,
            ],
        ]
    )


def matrix_to_euler(matrix_in: np.ndarray) -> tuple:
    matrix_in = np.array(matrix_in)
    """Convert 3x3 rotation matrix_in to euler angles in radians (ZYX order)."""
    if abs(matrix_in[2, 0]) >= 0.99999:  # Gimbal lock case
        yaw = math.atan2(-matrix_in[0, 1], matrix_in[1, 1])
        pitch = -math.pi / 2 if matrix_in[2, 0] > 0 else math.pi / 2
        roll = 0
    else:
        pitch = -math.asin(matrix_in[2, 0])
        roll = math.atan2(matrix_in[2, 1], matrix_in[2, 2])
        yaw = math.atan2(matrix_in[1, 0], matrix_in[0, 0])

    return (roll, pitch, yaw)


def matrix_to_quaternion(matrix_in: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix_in to quaternion (w,x,y,z)."""
    trace = np.trace(matrix_in)

    if trace > 0:
        S = math.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (matrix_in[2, 1] - matrix_in[1, 2]) / S
        y = (matrix_in[0, 2] - matrix_in[2, 0]) / S
        z = (matrix_in[1, 0] - matrix_in[0, 1]) / S
    elif (
        matrix_in[0, 0] > matrix_in[1, 1] and matrix_in[0, 0] > matrix_in[2, 2]
    ):
        S = (
            math.sqrt(
                1.0 + matrix_in[0, 0] - matrix_in[1, 1] - matrix_in[2, 2]
            )
            * 2
        )
        w = (matrix_in[2, 1] - matrix_in[1, 2]) / S
        x = 0.25 * S
        y = (matrix_in[0, 1] + matrix_in[1, 0]) / S
        z = (matrix_in[0, 2] + matrix_in[2, 0]) / S
    elif matrix_in[1, 1] > matrix_in[2, 2]:
        S = (
            math.sqrt(
                1.0 + matrix_in[1, 1] - matrix_in[0, 0] - matrix_in[2, 2]
            )
            * 2
        )
        w = (matrix_in[0, 2] - matrix_in[2, 0]) / S
        x = (matrix_in[0, 1] + matrix_in[1, 0]) / S
        y = 0.25 * S
        z = (matrix_in[1, 2] + matrix_in[2, 1]) / S
    else:
        S = (
            math.sqrt(
                1.0 + matrix_in[2, 2] - matrix_in[0, 0] - matrix_in[1, 1]
            )
            * 2
        )
        w = (matrix_in[1, 0] - matrix_in[0, 1]) / S
        x = (matrix_in[0, 2] + matrix_in[2, 0]) / S
        y = (matrix_in[1, 2] + matrix_in[2, 1]) / S
        z = 0.25 * S

    return np.array([w, x, y, z])


def degrees_to_radians(angles: tuple) -> tuple:
    return tuple(math.radians(angle) for angle in angles)


def radians_to_degrees(angles: tuple) -> tuple:
    return tuple(math.degrees(angle) for angle in angles)


def get_transform_matrix(input_order, output_order):
    """
    Generate a transformation matrix based on input and output coordinate specifications.

    Parameters:
    input_order (list): List of coordinates in input order, e.g. ['x', 'y', 'z']
    output_order (list): List of coordinates in output order with optional negation,
                        e.g. ['-x', 'z', 'y']

    Returns:
    numpy.ndarray: 3x3 transformation matrix
    """
    # Initialize zero matrix
    matrix = np.zeros((3, 3))

    # Map coordinate names to indices
    coord_to_idx = {"x": 0, "y": 1, "z": 2}

    # Process each output coordinate
    for out_idx, out_coord in enumerate(output_order):
        # Check for negation
        sign = -1 if out_coord.startswith("-") else 1
        # Remove negation sign if present
        coord = out_coord.replace("-", "")

        # Find which input coordinate this output corresponds to
        in_coord = input_order[coord_to_idx[coord]]
        in_idx = coord_to_idx[in_coord]

        # Set matrix element
        matrix[out_idx, in_idx] = sign

    return matrix


def transform_position(position, direction="sim_to_real"):
    """
    Transform 3D coordinates between simulation and real-world coordinate systems.

    Parameters:
    position (array-like): 3D position as [x, y, z]
    direction (str): Either 'sim_to_real' or 'real_to_sim'

    Returns:
    tuple: position transformed in specified format
    """
    # Convert inputs to numpy arrays
    position = np.array(position)

    # Create the position transformation matrix
    transform_matrix = get_transform_matrix(["x", "y", "z"], ["x", "z", "y"])

    if direction == "sim_to_real":
        # Transform position
        new_position = transform_matrix @ position

    elif direction == "real_to_sim":
        # Transform position (inverse transform)
        new_position = transform_matrix.T @ position

    return new_position


def transform_rotation(rotation, rotation_format="euler"):
    """
    Transform 3D coordinates between simulation and real-world coordinate systems.

    Parameters:
    rotation (array-like): 3D rotation as [roll, pitch, yaw] in radians
    rotation_format (str): Output rotation format - 'euler', 'matrix', or 'quaternion'

    Returns:
    tuple: rotation transformed in specified format
    """
    # Convert inputs to numpy arrays
    rotation = np.array(rotation)

    new_euler = np.array(
        [rotation[0], -rotation[2], -rotation[1]]
    )  # [roll, yaw, pitch]

    # Convert rotation to requested format
    if rotation_format.lower() == "euler":
        new_rotation = new_euler
    elif rotation_format.lower() == "matrix":
        new_rotation = euler_to_matrix(new_euler)
    elif rotation_format.lower() == "quaternion":
        new_rotation = euler_to_quaternion(new_euler)
    else:
        raise ValueError(
            "rotation_format must be 'euler', 'matrix', or 'quaternion'"
        )

    return new_rotation


def transform_3d_coordinates(
    position,
    rotation,
    direction="sim_to_real",
    rotation_format="euler",
):
    """
    Transform 3D coordinates between simulation and real-world coordinate systems.

    Parameters:
    position (array-like): 3D position as [x, y, z]
    rotation (array-like): 3D rotation as [roll, pitch, yaw] in radians
    direction (str): Either 'sim_to_real' or 'real_to_sim'
    rotation_format (str): Output rotation format - 'euler', 'matrix', or 'quaternion'

    Returns:
    tuple: (position, rotation) transformed coordinates, with rotation in specified format
    """
    new_position = transform_position(position, direction)
    new_rotation = transform_rotation(rotation, rotation_format)
    # Convert inputs to numpy arrays
    position = np.array(position)
    rotation = np.array(rotation)

    return new_position, new_rotation


def create_tform_matrix(translation, rotation, rotation_format="np"):
    # Create 4x4 matrix
    tform_matrix = np.eye(4)  # Create identity matrix

    # Set rotation part (upper-left 3x3)
    tform_matrix[:3, :3] = rotation

    # Set translation part (upper-right 3x1)
    tform_matrix[:3, 3] = translation
    if rotation_format == "mn":
        tform_matrix = mn.Matrix4(tform_matrix)

    return tform_matrix
