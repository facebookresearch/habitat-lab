#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import List, Optional, Tuple

import cv2
import imageio
import numpy as np
import scipy.ndimage

from habitat.core.simulator import Simulator
from habitat.utils.visualizations import utils

AGENT_SPRITE = imageio.imread(
    os.path.join(
        os.path.dirname(__file__),
        "assets",
        "maps_topdown_agent_sprite",
        "100x100.png",
    )
)
AGENT_SPRITE = np.ascontiguousarray(np.flipud(AGENT_SPRITE))
COORDINATE_EPSILON = 1e-6
COORDINATE_MIN = -62.3241 - COORDINATE_EPSILON
COORDINATE_MAX = 90.0399 + COORDINATE_EPSILON

MAP_INVALID_POINT = 0
MAP_VALID_POINT = 1
MAP_BORDER_INDICATOR = 2
MAP_SOURCE_POINT_INDICATOR = 4
MAP_TARGET_POINT_INDICATOR = 6
MAP_SHORTEST_PATH_COLOR = 7
TOP_DOWN_MAP_COLORS = np.full((256, 3), 150, dtype=np.uint8)
TOP_DOWN_MAP_COLORS[10:] = cv2.applyColorMap(
    np.arange(246, dtype=np.uint8), cv2.COLORMAP_JET
).squeeze(1)[:, ::-1]
TOP_DOWN_MAP_COLORS[MAP_INVALID_POINT] = [255, 255, 255]
TOP_DOWN_MAP_COLORS[MAP_VALID_POINT] = [150, 150, 150]
TOP_DOWN_MAP_COLORS[MAP_BORDER_INDICATOR] = [50, 50, 50]
TOP_DOWN_MAP_COLORS[MAP_SOURCE_POINT_INDICATOR] = [0, 0, 200]
TOP_DOWN_MAP_COLORS[MAP_TARGET_POINT_INDICATOR] = [200, 0, 0]
TOP_DOWN_MAP_COLORS[MAP_SHORTEST_PATH_COLOR] = [0, 200, 0]


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

    goal_distance_padding = max(
        2, 2 ** np.ceil(np.log(max(1e-6, goal_agent_dist)) / np.log(2))
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
    coordinate_min: float,
    coordinate_max: float,
    grid_resolution: Tuple[int, int],
) -> Tuple[int, int]:
    r"""Return gridworld index of realworld coordinates assuming top-left corner
    is the origin. The real world coordinates of lower left corner are
    (coordinate_min, coordinate_min) and of top right corner are
    (coordinate_max, coordinate_max)
    """
    grid_size = (
        (coordinate_max - coordinate_min) / grid_resolution[0],
        (coordinate_max - coordinate_min) / grid_resolution[1],
    )
    grid_x = int((coordinate_max - realworld_x) / grid_size[0])
    grid_y = int((realworld_y - coordinate_min) / grid_size[1])
    return grid_x, grid_y


def from_grid(
    grid_x: int,
    grid_y: int,
    coordinate_min: float,
    coordinate_max: float,
    grid_resolution: Tuple[int, int],
) -> Tuple[float, float]:
    r"""Inverse of _to_grid function. Return real world coordinate from
    gridworld assuming top-left corner is the origin. The real world
    coordinates of lower left corner are (coordinate_min, coordinate_min) and
    of top right corner are (coordinate_max, coordinate_max)
    """
    grid_size = (
        (coordinate_max - coordinate_min) / grid_resolution[0],
        (coordinate_max - coordinate_min) / grid_resolution[1],
    )
    realworld_x = coordinate_max - grid_x * grid_size[0]
    realworld_y = coordinate_min + grid_y * grid_size[1]
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


def get_topdown_map(
    sim: Simulator,
    map_resolution: Tuple[int, int] = (1250, 1250),
    num_samples: int = 20000,
    draw_border: bool = True,
) -> np.ndarray:
    r"""Return a top-down occupancy map for a sim. Note, this only returns valid
    values for whatever floor the agent is currently on.

    Args:
        sim: The simulator.
        map_resolution: The resolution of map which will be computed and
            returned.
        num_samples: The number of random navigable points which will be
            initially
            sampled. For large environments it may need to be increased.
        draw_border: Whether to outline the border of the occupied spaces.

    Returns:
        Image containing 0 if occupied, 1 if unoccupied, and 2 if border (if
        the flag is set).
    """
    top_down_map = np.zeros(map_resolution, dtype=np.uint8)
    border_padding = 3

    start_height = sim.get_agent_state().position[1]

    # Use sampling to find the extrema points that might be navigable.
    range_x = (map_resolution[0], 0)
    range_y = (map_resolution[1], 0)
    for _ in range(num_samples):
        point = sim.sample_navigable_point()
        # Check if on same level as original
        if np.abs(start_height - point[1]) > 0.5:
            continue
        g_x, g_y = to_grid(
            point[0], point[2], COORDINATE_MIN, COORDINATE_MAX, map_resolution
        )
        range_x = (min(range_x[0], g_x), max(range_x[1], g_x))
        range_y = (min(range_y[0], g_y), max(range_y[1], g_y))

    # Pad the range just in case not enough points were sampled to get the true
    # extrema.
    padding = int(np.ceil(map_resolution[0] / 125))
    range_x = (
        max(range_x[0] - padding, 0),
        min(range_x[-1] + padding + 1, top_down_map.shape[0]),
    )
    range_y = (
        max(range_y[0] - padding, 0),
        min(range_y[-1] + padding + 1, top_down_map.shape[1]),
    )

    # Search over grid for valid points.
    for ii in range(range_x[0], range_x[1]):
        for jj in range(range_y[0], range_y[1]):
            realworld_x, realworld_y = from_grid(
                ii, jj, COORDINATE_MIN, COORDINATE_MAX, map_resolution
            )
            valid_point = sim.is_navigable(
                [realworld_x, start_height, realworld_y]
            )
            top_down_map[ii, jj] = (
                MAP_VALID_POINT if valid_point else MAP_INVALID_POINT
            )

    # Draw border if necessary
    if draw_border:
        # Recompute range in case padding added any more values.
        range_x = np.where(np.any(top_down_map, axis=1))[0]
        range_y = np.where(np.any(top_down_map, axis=0))[0]
        range_x = (
            max(range_x[0] - border_padding, 0),
            min(range_x[-1] + border_padding + 1, top_down_map.shape[0]),
        )
        range_y = (
            max(range_y[0] - border_padding, 0),
            min(range_y[-1] + border_padding + 1, top_down_map.shape[1]),
        )

        _outline_border(
            top_down_map[range_x[0] : range_x[1], range_y[0] : range_y[1]]
        )
    return top_down_map


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
    path_points: List[Tuple],
    color: int,
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
        cv2.line(top_down_map, prev_pt, next_pt, color, thickness=thickness)
