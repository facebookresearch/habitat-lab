import numpy as np

from habitat.utils.geometry_utils import quaternion_from_two_vectors


def direction_to_quaternion(direction_vector: np.ndarray):
    """
    Convert a direction vector to a quaternion.
    """
    origin_vector = np.array([0, 0, -1])
    output = quaternion_from_two_vectors(origin_vector, direction_vector)
    output = output.normalized()
    return output
