#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio
import numpy as np
import scipy.ndimage

from habitat.utils.visualizations import utils

try:
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
except ImportError:
    pass

import cv2

AGENT_SPRITE = imageio.imread(
    os.path.join(
        os.path.dirname(__file__),
        "assets",
        "maps_topdown_agent_sprite",
        "100x100.png",
    )
)
AGENT_SPRITE = np.ascontiguousarray(np.flipud(AGENT_SPRITE))

MAP_INVALID_POINT = 0
MAP_VALID_POINT = 1
MAP_BORDER_INDICATOR = 2
MAP_SOURCE_POINT_INDICATOR = 4
MAP_TARGET_POINT_INDICATOR = 6
MAP_SHORTEST_PATH_COLOR = 7
MAP_VIEW_POINT_INDICATOR = 8
MAP_TARGET_BOUNDING_BOX = 9
TOP_DOWN_MAP_COLORS = np.full((256, 3), 150, dtype=np.uint8)
TOP_DOWN_MAP_COLORS[10:] = cv2.applyColorMap(
    np.arange(246, dtype=np.uint8), cv2.COLORMAP_JET
).squeeze(1)[:, ::-1]
TOP_DOWN_MAP_COLORS[MAP_INVALID_POINT] = [255, 255, 255]  # White
TOP_DOWN_MAP_COLORS[MAP_VALID_POINT] = [150, 150, 150]  # Light Grey
TOP_DOWN_MAP_COLORS[MAP_BORDER_INDICATOR] = [50, 50, 50]  # Grey
TOP_DOWN_MAP_COLORS[MAP_SOURCE_POINT_INDICATOR] = [0, 0, 200]  # Blue
TOP_DOWN_MAP_COLORS[MAP_TARGET_POINT_INDICATOR] = [200, 0, 0]  # Red
TOP_DOWN_MAP_COLORS[MAP_SHORTEST_PATH_COLOR] = [0, 200, 0]  # Green
TOP_DOWN_MAP_COLORS[MAP_VIEW_POINT_INDICATOR] = [245, 150, 150]  # Light Red
TOP_DOWN_MAP_COLORS[MAP_TARGET_BOUNDING_BOX] = [0, 175, 0]  # Green


def draw_agent(
    image: np.ndarray,
    agent_center_coord: Tuple[int, int],
    agent_rotation: float,
    agent_radius_px: int = 5,
) -> np.ndarray:
    r"""Return an image with the agent image composited onto it.
    Args:
        image: the image onto which to put the agent.
        agent_center_coord: the image coordinates where to paste the agent.
        agent_rotation: the agent's current rotation in radians.
        agent_radius_px: 1/2 number of pixels the agent will be resized to.
    Returns:
        The modified background image. This operation is in place.
    """

    # Rotate before resize to keep good resolution.
    rotated_agent = scipy.ndimage.interpolation.rotate(
        AGENT_SPRITE, agent_rotation * 180 / np.pi
    )
    # Rescale because rotation may result in larger image than original, but
    # the agent sprite size should stay the same.
    initial_agent_size = AGENT_SPRITE.shape[0]
    new_size = rotated_agent.shape[0]
    agent_size_px = max(
        1, int(agent_radius_px * 2 * new_size / initial_agent_size)
    )
    resized_agent = cv2.resize(
        rotated_agent,
        (agent_size_px, agent_size_px),
        interpolation=cv2.INTER_LINEAR,
    )
    utils.paste_overlapping_image(image, resized_agent, agent_center_coord)
    return image


def pointnav_draw_target_birdseye_view(
    agent_position: np.ndarray,
    agent_heading: float,
    goal_position: np.ndarray,
    resolution_px: int = 800,
    goal_radius: float = 0.2,
    agent_radius_px: int = 20,
    target_band_radii: Optional[List[float]] = None,
    target_band_colors: Optional[List[Tuple[int, int, int]]] = None,
) -> np.ndarray:
    r"""Return an image of agent w.r.t. centered target location for pointnav
    tasks.

    Args:
        agent_position: the agent's current position.
        agent_heading: the agent's current rotation in radians. This can be
            found using the HeadingSensor.
        goal_position: the pointnav task goal position.
        resolution_px: number of pixels for the output image width and height.
        goal_radius: how near the agent needs to be to be successful for the
            pointnav task.
        agent_radius_px: 1/2 number of pixels the agent will be resized to.
        target_band_radii: distance in meters to the outer-radius of each band
            in the target image.
        target_band_colors: colors in RGB 0-255 for the bands in the target.
    Returns:
        Image centered on the goal with the agent's current relative position
        and rotation represented by an arrow. To make the rotations align
        visually with habitat, positive-z is up, positive-x is left and a
        rotation of 0 points upwards in the output image and rotates clockwise.
    """
    if target_band_radii is None:
        target_band_radii = [20, 10, 5, 2.5, 1]
    if target_band_colors is None:
        target_band_colors = [
            (47, 19, 122),
            (22, 99, 170),
            (92, 177, 0),
            (226, 169, 0),
            (226, 12, 29),
        ]

    assert len(target_band_radii) == len(
        target_band_colors
    ), "There must be an equal number of scales and colors."

    goal_agent_dist = np.linalg.norm(agent_position - goal_position, 2)

    goal_distance_padding = np.maximum(
        2, 2 ** np.ceil(np.log(np.maximum(1e-6, goal_agent_dist)) / np.log(2))
    )
    movement_scale = 1.0 / goal_distance_padding
    half_res = resolution_px // 2
    im_position = np.full(
        (resolution_px, resolution_px, 3), 255, dtype=np.uint8
    )

    # Draw bands:
    for scale, color in zip(target_band_radii, target_band_colors):
        if goal_distance_padding * 4 > scale:
            cv2.circle(
                im_position,
                (half_res, half_res),
                max(2, int(half_res * scale * movement_scale)),
                color,
                thickness=-1,
            )

    # Draw such that the agent being inside the radius is the circles
    # overlapping.
    cv2.circle(
        im_position,
        (half_res, half_res),
        max(2, int(half_res * goal_radius * movement_scale)),
        (127, 0, 0),
        thickness=-1,
    )

    relative_position = agent_position - goal_position
    # swap x and z, remove y for (x,y,z) -> image coordinates.
    relative_position = relative_position[[2, 0]]
    relative_position *= half_res * movement_scale
    relative_position += half_res
    relative_position = np.round(relative_position).astype(np.int32)

    # Draw the agent
    draw_agent(im_position, relative_position, agent_heading, agent_radius_px)

    # Rotate twice to fix coordinate system to upwards being positive-z.
    # Rotate instead of flip to keep agent rotations in sync with egocentric
    # view.
    im_position = np.rot90(im_position, 2)
    return im_position


def to_grid(
    realworld_x: float,
    realworld_y: float,
    grid_resolution: Tuple[int, int],
    sim: Optional["HabitatSim"] = None,
    pathfinder=None,
) -> Tuple[int, int]:
    r"""Return gridworld index of realworld coordinates assuming top-left corner
    is the origin. The real world coordinates of lower left corner are
    (coordinate_min, coordinate_min) and of top right corner are
    (coordinate_max, coordinate_max)
    """
    if sim is None and pathfinder is None:
        raise RuntimeError(
            "Must provide either a simulator or pathfinder instance"
        )

    if pathfinder is None:
        pathfinder = sim.pathfinder

    lower_bound, upper_bound = pathfinder.get_bounds()

    grid_size = (
        abs(upper_bound[2] - lower_bound[2]) / grid_resolution[0],
        abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
    )
    grid_x = int((realworld_x - lower_bound[2]) / grid_size[0])
    grid_y = int((realworld_y - lower_bound[0]) / grid_size[1])
    return grid_x, grid_y


def from_grid(
    grid_x: int,
    grid_y: int,
    grid_resolution: Tuple[int, int],
    sim: Optional["HabitatSim"] = None,
    pathfinder=None,
) -> Tuple[float, float]:
    r"""Inverse of _to_grid function. Return real world coordinate from
    gridworld assuming top-left corner is the origin. The real world
    coordinates of lower left corner are (coordinate_min, coordinate_min) and
    of top right corner are (coordinate_max, coordinate_max)
    """

    if sim is None and pathfinder is None:
        raise RuntimeError(
            "Must provide either a simulator or pathfinder instance"
        )

    if pathfinder is None:
        pathfinder = sim.pathfinder

    lower_bound, upper_bound = pathfinder.get_bounds()

    grid_size = (
        abs(upper_bound[2] - lower_bound[2]) / grid_resolution[0],
        abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
    )
    realworld_x = lower_bound[2] + grid_x * grid_size[0]
    realworld_y = lower_bound[0] + grid_y * grid_size[1]
    return realworld_x, realworld_y


def _outline_border(top_down_map):
    left_right_block_nav = (top_down_map[:, :-1] == 1) & (
        top_down_map[:, :-1] != top_down_map[:, 1:]
    )
    left_right_nav_block = (top_down_map[:, 1:] == 1) & (
        top_down_map[:, :-1] != top_down_map[:, 1:]
    )

    up_down_block_nav = (top_down_map[:-1] == 1) & (
        top_down_map[:-1] != top_down_map[1:]
    )
    up_down_nav_block = (top_down_map[1:] == 1) & (
        top_down_map[:-1] != top_down_map[1:]
    )

    top_down_map[:, :-1][left_right_block_nav] = MAP_BORDER_INDICATOR
    top_down_map[:, 1:][left_right_nav_block] = MAP_BORDER_INDICATOR

    top_down_map[:-1][up_down_block_nav] = MAP_BORDER_INDICATOR
    top_down_map[1:][up_down_nav_block] = MAP_BORDER_INDICATOR


def calculate_meters_per_pixel(
    map_resolution: int, sim: Optional["HabitatSim"] = None, pathfinder=None
):
    r"""Calculate the meters_per_pixel for a given map resolution"""
    if sim is None and pathfinder is None:
        raise RuntimeError(
            "Must provide either a simulator or pathfinder instance"
        )

    if pathfinder is None:
        pathfinder = sim.pathfinder

    lower_bound, upper_bound = pathfinder.get_bounds()
    return min(
        abs(upper_bound[coord] - lower_bound[coord]) / map_resolution
        for coord in [0, 2]
    )


def get_topdown_map(
    pathfinder,
    height: float,
    map_resolution: int = 1024,
    draw_border: bool = True,
    meters_per_pixel: Optional[float] = None,
) -> np.ndarray:
    r"""Return a top-down occupancy map for a sim. Note, this only returns valid
    values for whatever floor the agent is currently on.

    :param pathfinder: A habitat-sim pathfinder instances to get the map from
    :param height: The height in the environment to make the topdown map
    :param map_resolution: Length of the longest side of the map.  Used to calculate :p:`meters_per_pixel`
    :param draw_border: Whether or not to draw a border
    :param meters_per_pixel: Overrides map_resolution an

    :return: Image containing 0 if occupied, 1 if unoccupied, and 2 if border (if
        the flag is set).
    """

    if meters_per_pixel is None:
        meters_per_pixel = calculate_meters_per_pixel(
            map_resolution, pathfinder=pathfinder
        )

    top_down_map = pathfinder.get_topdown_view(
        meters_per_pixel=meters_per_pixel, height=height
    ).astype(np.uint8)

    # Draw border if necessary
    if draw_border:
        _outline_border(top_down_map)

    return np.ascontiguousarray(top_down_map)


def get_topdown_map_from_sim(
    sim: "HabitatSim",
    map_resolution: int = 1024,
    draw_border: bool = True,
    meters_per_pixel: Optional[float] = None,
    agent_id: int = 0,
) -> np.ndarray:
    r"""Wrapper around :py:`get_topdown_map` that retrieves that pathfinder and heigh from the current simulator

    :param sim: Simulator instance.
    :param agent_id: The agent ID
    """
    return get_topdown_map(
        sim.pathfinder,
        sim.get_agent(agent_id).state.position[1],
        map_resolution,
        draw_border,
        meters_per_pixel,
    )


def colorize_topdown_map(
    top_down_map: np.ndarray,
    fog_of_war_mask: Optional[np.ndarray] = None,
    fog_of_war_desat_amount: float = 0.5,
) -> np.ndarray:
    r"""Convert the top down map to RGB based on the indicator values.
    Args:
        top_down_map: A non-colored version of the map.
        fog_of_war_mask: A mask used to determine which parts of the
            top_down_map are visible
            Non-visible parts will be desaturated
        fog_of_war_desat_amount: Amount to desaturate the color of unexplored areas
            Decreasing this value will make unexplored areas darker
            Default: 0.5
    Returns:
        A colored version of the top-down map.
    """
    _map = TOP_DOWN_MAP_COLORS[top_down_map]

    if fog_of_war_mask is not None:
        fog_of_war_desat_values = np.array([[fog_of_war_desat_amount], [1.0]])
        # Only desaturate things that are valid points as only valid points get revealed
        desat_mask = top_down_map != MAP_INVALID_POINT

        _map[desat_mask] = (
            _map * fog_of_war_desat_values[fog_of_war_mask]
        ).astype(np.uint8)[desat_mask]

    return _map


def draw_path(
    top_down_map: np.ndarray,
    path_points: Sequence[Tuple],
    color: int = 10,
    thickness: int = 2,
) -> None:
    r"""Draw path on top_down_map (in place) with specified color.
    Args:
        top_down_map: A colored version of the map.
        color: color code of the path, from TOP_DOWN_MAP_COLORS.
        path_points: list of points that specify the path to be drawn
        thickness: thickness of the path.
    """
    for prev_pt, next_pt in zip(path_points[:-1], path_points[1:]):
        # Swapping x y
        cv2.line(
            top_down_map,
            prev_pt[::-1],
            next_pt[::-1],
            color,
            thickness=thickness,
        )  # type: ignore


def colorize_draw_agent_and_fit_to_height(
    topdown_map_info: Dict[str, Any], output_height: int
):
    r"""Given the output of the TopDownMap measure, colorizes the map, draws the agent,
    and fits to a desired output height

    :param topdown_map_info: The output of the TopDownMap measure
    :param output_height: The desired output height
    """
    top_down_map = topdown_map_info["map"]
    top_down_map = colorize_topdown_map(
        top_down_map, topdown_map_info["fog_of_war_mask"]
    )
    for agent_idx in range(len(topdown_map_info["agent_map_coord"])):
        map_agent_pos = topdown_map_info["agent_map_coord"][agent_idx]
        map_agent_angle = topdown_map_info["agent_angle"][agent_idx]
        top_down_map = draw_agent(
            image=top_down_map,
            agent_center_coord=map_agent_pos,
            agent_rotation=map_agent_angle,
            agent_radius_px=min(top_down_map.shape[0:2]) // 32,
        )

    if top_down_map.shape[0] > top_down_map.shape[1]:
        top_down_map = np.rot90(top_down_map, 1)

    # scale top down map to align with rgb view
    old_h, old_w, _ = top_down_map.shape
    top_down_height = output_height
    top_down_width = int(float(top_down_height) / old_h * old_w)
    # cv2 resize (dsize is width first)
    top_down_map = cv2.resize(
        top_down_map,
        (top_down_width, top_down_height),
        interpolation=cv2.INTER_CUBIC,
    )

    return top_down_map
