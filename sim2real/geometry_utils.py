import math
from typing import List, Tuple, Union

import magnum as mn
import numpy as np

import habitat_sim


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


def set_robot_base_transform(robot_id, global_T_base_std: mn.Matrix4) -> None:
    """
    Sets the robot's base's transform to the input transform. The input transform
    assumes that the base frame is oriented upright, however the URDF of Spot has
    been modified so that it is actually rolled +90 degrees. Thus, we need to rotate the
    input transform before it can be used to set the robot URDF base's transform. This
    rotation will be made around the robot's local forward (+x) axis for -90 degrees.

    Args:
        robot_id: ID of the articulated robot object
        transform: Magnum Matrix4 in standard (non-Habitat) convention
    """
    roll_offset = mn.Matrix3x3(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        )
    )
    global_T_base_raw_std = mn.Matrix4().from_(
        rotation_scaling=global_T_base_std.rotation() @ roll_offset,
        translation=global_T_base_std.translation,
    )
    global_T_base_raw_std.translation += mn.Vector3(0, 0, 0.48)

    new_tform = convert_conventions(global_T_base_raw_std, reverse=True)
    robot_id.transformation = new_tform
    # robot_id.sim_obj.transformation = new_tform


def get_robot_base_transform(robot_id) -> mn.Matrix4:
    """
    Get the robot base's transform, assuming that the identity rotation makes it upright
    in standard coordinate conventions. Since Spot's URDF has been modified so that it
    is actually rolled +90 degrees, when the roll is 0, it looks like +90. Thus, we must
    add a rotation around the robot's local forward (+x) axis for +90 degrees to the
    reading given by this function. This ensures that when the robot looks like it's at
    +90 roll, it also reads a roll of +90 (not 0).

    Args:
        robot_id: ID of the articulated robot object
    """
    roll_offset = mn.Matrix3x3(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        ).T  # transposed for inverse
    )
    global_T_base_raw_std = convert_conventions(robot_id.transformation)
    global_T_base_std = mn.Matrix4().from_(
        rotation_scaling=global_T_base_raw_std.rotation() @ roll_offset,
        translation=global_T_base_raw_std.translation,
    )

    return global_T_base_std


def get_ee_transform(robot_id) -> mn.Matrix4:
    # TODO (START): Use habitat-lab SpotRobot to get global_T_ee_raw_hab
    global_T_ee_raw_hab = robot_id.get_link_scene_node(0).transformation
    # global_T_ee_raw_hab = robot_id.sim_obj.get_link_scene_node(
    #     7
    # ).transformation
    global_T_ee_raw_hab.translation = global_T_ee_raw_hab.transform_point(
        mn.Vector3(0.08, 0, 0)
    )
    # TODO (END): Use habitat-lab SpotRobot to get global_T_ee_raw_hab

    global_T_ee_raw_std = convert_conventions(global_T_ee_raw_hab)
    global_T_ee_std = global_T_ee_raw_std
    return global_T_ee_std


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


def generate_random_quaternion():
    # Generate random scalar (w) between -1 and 1
    w = np.random.uniform(-1, 1)

    # Generate random vector (x, y, z)
    v = np.random.uniform(-1, 1, 3)

    # Normalize to ensure w² + x² + y² + z² = 1
    norm = np.sqrt(w * w + np.sum(v * v))
    w /= norm
    v /= norm

    return w, v


def constrain_quaternion(q_a, q_b, max_angle):
    """
    Calculate quaternion C that is at most max_angle radians from q_a,
    and equals q_b if within that constraint.

    Args:
        q_a: Starting quaternion as (w, [x, y, z])
        q_b: Target quaternion as (w, [x, y, z])
        max_angle: Maximum allowed angle in radians

    Returns:
        Tuple (w, [x, y, z]) representing the constrained quaternion
    """
    w_a, v_a = q_a
    w_b, v_b = q_b

    # Calculate cosine of angle between quaternions
    dot_product = w_a * w_b + np.dot(v_a, v_b)
    angle = np.arccos(min(abs(dot_product), 1.0)) * 2

    # If within max_angle, return q_b
    if angle <= max_angle:
        return q_b

    # Otherwise, interpolate to max allowed angle
    t = max_angle / angle

    # Handle negative dot product by negating q_b
    if dot_product < 0:
        w_b = -w_b
        v_b = -v_b

    # Spherical linear interpolation (SLERP)
    sin_angle = np.sin(angle)
    if sin_angle == 0:
        return q_a

    w_c = (w_a * np.sin((1 - t) * angle) + w_b * np.sin(t * angle)) / sin_angle
    v_c = (v_a * np.sin((1 - t) * angle) + v_b * np.sin(t * angle)) / sin_angle

    # Normalize to ensure unit quaternion
    norm = np.sqrt(w_c * w_c + np.sum(v_c * v_c))
    return w_c / norm, v_c / norm


def magnum_to_agent_state(
    transformation: mn.Matrix4,
) -> habitat_sim.AgentState:
    mn_quaternion = mn.Quaternion().from_matrix(transformation.rotation())
    agent_state = habitat_sim.AgentState()
    agent_state.position = transformation.translation
    agent_state.rotation = [*mn_quaternion.vector, mn_quaternion.scalar]
    return agent_state


def generate_matrices():
    """
    Generates all possible 3x3 matrices that:
    - Contain exactly three non-zero elements (1 or -1)
    - Have at most one non-zero element per row and column

    Returns:
        list: List of numpy arrays representing valid matrices
    """
    from itertools import permutations, product

    # First, let's generate all possible positions for the three ones
    # We need exactly one non-zero element per row and column
    valid_matrices = []

    # Generate all possible row indices (0,1,2) for the three columns
    possible_positions = list(permutations(range(3), 3))

    # For each valid position combination, we'll try all possible combinations
    # of positive and negative ones (2^3 = 8 possibilities)
    sign_combinations = list(product([-1, 1], repeat=3))

    for positions in possible_positions:
        for signs in sign_combinations:
            # Create empty 3x3 matrix
            matrix = np.zeros((3, 3))

            # Fill in the non-zero elements
            for col, (row, sign) in enumerate(zip(positions, signs)):
                matrix[row, col] = sign

            valid_matrices.append(matrix)

    return valid_matrices


def generate_random_transform():
    """
    Generate a random 4x4 homogeneous transformation matrix.

    Returns:
        numpy.ndarray: A 4x4 homogeneous transformation matrix where:
            - Top-left 3x3 is a random rotation matrix
            - Top-right 3x1 is a random translation vector
            - Bottom row is [0, 0, 0, 1] to maintain homogeneous form
    """
    # Generate a random 3x3 rotation matrix using QR decomposition
    # This ensures the matrix is orthogonal (valid rotation)
    random_matrix = np.random.rand(3, 3)
    q, r = np.linalg.qr(random_matrix)
    rotation = q * np.sign(np.diag(r))[:, None]  # Ensure determinant is 1

    # Generate random translation vector
    translation = np.random.uniform(-10, 10, (3, 1))

    # Create the 4x4 homogeneous transformation matrix
    transform = np.zeros((4, 4))
    transform[:3, :3] = rotation
    transform[:3, 3:] = translation
    transform[3, 3] = 1.0

    return transform


def generate_circle_points(
    radius: float, num_points: int
) -> List[Tuple[float, float]]:
    """
    Generate points along a circle with given radius and number of points.
    Points start at (0, radius) and go clockwise.

    Args:
        radius (float): Radius of the circle
        num_points (int): Number of points to generate

    Returns:
        List[Tuple[float, float]]: List of (x, y) coordinates
    """
    points = []
    angle_step = 2 * math.pi / num_points

    for i in range(num_points):
        angle = (math.pi / 2) - (
            i * angle_step
        )  # Start at (0, r) and go clockwise
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        points.append(
            (round(x, 6), round(y, 6))
        )  # Round to avoid floating point imprecision

    return points
