import math
import os

import cv2
import imageio
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


def save_viewpoint_frame(obs, obj_handle, obj_semantic_id, act_idx):
    rgb_obs = np.ascontiguousarray(obs["color"][..., :3])
    sem_obs = (obs["semantic"] == obj_semantic_id).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        sem_obs, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )
    rgb_obs = cv2.drawContours(rgb_obs, contours, -1, (0, 255, 0), 4)
    img_dir = "data/images/objnav_dataset_gen"
    os.makedirs(img_dir, exist_ok=True)
    imageio.imsave(
        os.path.join(
            img_dir,
            f"{obj_handle}_{obj_semantic_id}_{x}_{z}_{act_idx}.png",
        ),
        rgb_obs,
    )


def save_topdown_map(
    sim,
    view_locations,
    candidate_poses_ious_orig,
    poses_type_counter,
    object_handle,
    object_position,
    object_aabb,
    object_semantic_id,
    island_limit_radius,
):
    object_position_on_floor = np.array(object_position).copy()
    if len(view_locations) > 0:
        object_position_on_floor[1] = view_locations[0].agent_state.position[1]
    else:
        while True:
            # TODO (Mukul): think of better way than ->
            pf = sim.pathfinder
            navigable_point = pf.get_random_navigable_point()
            if pf.island_radius(navigable_point) >= island_limit_radius:
                break

        object_position_on_floor[1] = navigable_point[1]

    topdown_map = get_topdown_map(
        sim, start_pos=object_position_on_floor, marker=None
    )

    colors = list(COLOR_PALETTE.values())
    for p in candidate_poses_ious_orig:
        if p[0] < 0:
            color = colors[p[-1].value - 1]
        else:
            color = COLOR_PALETTE["green"]

        topdown_map = get_topdown_map(
            sim,
            start_pos=p[1],
            topdown_map=topdown_map,
            marker="circle",
            radius=2,
            color=color,
        )

    topdown_map = get_topdown_map(
        sim,
        start_pos=object_position_on_floor,
        topdown_map=topdown_map,
        marker="circle",
        radius=6,
        color=COLOR_PALETTE["red"],
    )
    topdown_map = draw_obj_bbox_on_topdown_map(topdown_map, object_aabb, sim)

    if len(view_locations) == 0:
        h = topdown_map.shape[0]
        topdown_map = cv2.copyMakeBorder(
            topdown_map,
            0,
            150,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=COLOR_PALETTE["white"],
        )

        for i, c in enumerate(poses_type_counter.items()):
            line = f"{c[0].name}: {c[1]}/{len(candidate_poses_ious_orig)}"
            color = colors[c[0].value - 1]
            topdown_map = cv2.putText(
                topdown_map,
                line,
                (10, h + 25 * i + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    img_dir = "data/images/objnav_dataset_gen/maps"
    os.makedirs(img_dir, exist_ok=True)
    imageio.imsave(
        os.path.join(img_dir, f"{object_handle}_{object_semantic_id}.png"),
        topdown_map,
    )


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
