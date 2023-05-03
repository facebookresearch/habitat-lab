import math

import cv2
import magnum as mn
import numpy as np

import habitat_sim
from habitat.utils.visualizations import maps
from habitat_sim.utils import common as utils

COLOR_PALETTE = {
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "lighter_blue": (0, 100, 255),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "pink": (255, 0, 127),
    "green": (0, 255, 0),
}


def get_topdown_map(
    sim,
    start_pos=None,
    start_rot=None,
    topdown_map=None,
    marker="sprite",
    color=COLOR_PALETTE["blue"],
    radius=8,
    boundary=False,
):
    if start_pos is None:
        start_pos = sim.get_agent(0).get_state().position
    if start_rot is None:
        start_rot = sim.get_agent(0).get_state().rotation
    if topdown_map is None:
        topdown_map = maps.get_topdown_map(sim.pathfinder, height=start_pos[1])

        recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        )
        topdown_map = recolor_map[topdown_map]

    grid_dimensions = (topdown_map.shape[0], topdown_map.shape[1])

    # convert world agent position to maps module grid point
    agent_grid_pos_source = maps.to_grid(
        start_pos[2], start_pos[0], grid_dimensions, pathfinder=sim.pathfinder
    )

    agent_grid_pos_source = maps.to_grid(
        start_pos[2], start_pos[0], grid_dimensions, pathfinder=sim.pathfinder
    )
    agent_forward = utils.quat_to_magnum(
        sim.get_agent(0).get_state().rotation
    ).transform_vector(mn.Vector3(0, 0, -1.0))
    agent_forward = utils.quat_to_magnum(
        sim.agents[0].get_state().rotation
    ).transform_vector(mn.Vector3(0, 0, -1.0))

    agent_orientation = math.atan2(agent_forward[0], agent_forward[2])
    if marker == "sprite":
        maps.draw_agent(
            topdown_map,
            agent_grid_pos_source,
            agent_orientation,
            agent_radius_px=24,
        )
    elif marker == "circle":
        topdown_map = cv2.circle(
            topdown_map,
            agent_grid_pos_source[::-1],
            radius=radius,
            color=color,
            thickness=-1,
        )

        if boundary:
            topdown_map = cv2.circle(
                topdown_map,
                agent_grid_pos_source[::-1],
                radius=radius + 2,
                color=COLOR_PALETTE["white"],
                thickness=1,
            )

    return topdown_map


def get_topdown_map_with_path(sim, start_pos, start_rot, goal_pos):
    topdown_map = get_topdown_map(sim, start_pos, start_rot)
    grid_dimensions = (topdown_map.shape[0], topdown_map.shape[1])

    path = habitat_sim.ShortestPath()
    path.requested_start = start_pos
    path.requested_end = goal_pos

    assert sim.pathfinder.find_path(path)

    # print(f"geodesic_distance: {path.geodesic_distance})

    trajectory = [
        maps.to_grid(
            path_point[2],
            path_point[0],
            grid_dimensions,
            pathfinder=sim.pathfinder,
        )
        for path_point in path.points
    ]
    # draw the trajectory on the map
    maps.draw_path(topdown_map, trajectory)

    return topdown_map


def draw_obj_bbox_on_topdown_map(topdown_map, object_aabb, sim):
    object_topdown_bb_corner_1 = np.array(object_aabb.back_top_left)
    object_topdown_bb_corner_2 = np.array(object_aabb.front_top_right)
    grid_dimensions = (topdown_map.shape[0], topdown_map.shape[1])
    object_topdown_bb_corner_1 = maps.to_grid(
        object_topdown_bb_corner_1[2],
        object_topdown_bb_corner_1[0],
        grid_dimensions,
        pathfinder=sim.pathfinder,
    )
    object_topdown_bb_corner_2 = maps.to_grid(
        object_topdown_bb_corner_2[2],
        object_topdown_bb_corner_2[0],
        grid_dimensions,
        pathfinder=sim.pathfinder,
    )
    topdown_map = cv2.rectangle(
        topdown_map,
        object_topdown_bb_corner_1[::-1],
        object_topdown_bb_corner_2[::-1],
        color=COLOR_PALETTE["black"],
        thickness=2,
    )
    return topdown_map
