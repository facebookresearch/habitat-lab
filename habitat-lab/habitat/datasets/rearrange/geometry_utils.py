import numpy as np
import magnum as mn

from habitat.utils.geometry_utils import quaternion_from_two_vectors

def get_bb(obj):
    obj_node = obj.root_scene_node
    obj_bb = obj_node.cumulative_bb
    corners = get_corners(obj_bb, obj_node)
    tranformed_bb = get_bbs_from_corners(corners)
    return tranformed_bb

def get_corners(obj_bb, obj_node=None):
    corners = ["back_top_right", "back_top_left", "back_bottom_right", "back_bottom_left",
               "front_top_right", "front_top_left", "front_bottom_right", "front_bottom_left"]
    surface_corners = [getattr(obj_bb, cor) for cor in corners]

    if obj_node is not None:
        surface_corners = [obj_node.transformation.transform_point(cor) for cor in surface_corners]

    return surface_corners

def get_bbs_from_corners(cors):
    min_x, min_y, min_z = [1e3] * 3
    max_x, max_y, max_z = [-1e3] * 3

    for cor in cors:
        max_x = cor.x if cor.x > max_x else max_x
        max_y = cor.y if cor.y > max_y else max_y
        max_z = cor.z if cor.z > max_z else max_z

        min_x = cor.x if cor.x < min_x else min_x
        min_y = cor.y if cor.y < min_y else min_y
        min_z = cor.z if cor.z < min_z else min_z

    min_coords = mn.Vector3(min_x, min_y, min_z)
    size = mn.Vector3(max_x - min_x, max_y - min_y, max_z - min_z)
    bb = mn.Range3D.from_size(min_coords, size)
    return bb

def direction_to_quaternion(direction_vector: np.array):
    origin_vector = np.array([0, 0, -1])
    output = quaternion_from_two_vectors(origin_vector, direction_vector)
    output = output.normalized()
    return output
