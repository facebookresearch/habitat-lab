#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import cv2
import scipy.misc
import scipy.ndimage
import os
from typing import List, Tuple, Optional

AGENT_SPRITE = scipy.misc.imread(
    os.path.join(
        os.path.dirname(__file__),
        "assets",
        "maps_topdown_agent_sprite",
        "100x100.png",
    )
)
AGENT_SPRITE = np.ascontiguousarray(np.flipud(AGENT_SPRITE))


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

    """Return an image of agent w.r.t. centered target location for pointnav
    tasks.

    Args:
        agent_position: the agent's current position.
        agent_heading: the agent's current rotation in radians. This can be
            found using the HeadingSensor.
        goal_position: the pointnav task goal position.
        resolution_px: number of pixels for the output image width and height.
        goal_radius: how near the agent needs to be to be successful for the
            pointnav task.
        agent_radius_px: number of pixels the agent will be
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
        2, 2 ** np.ceil(np.log(goal_agent_dist) / np.log(2))
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

    # Rotate before resize to keep good resolution.
    rotated_agent = scipy.ndimage.interpolation.rotate(
        AGENT_SPRITE, agent_heading * -180 / np.pi
    )
    # Rescale because rotation may result in larger image than original, but the
    # agent sprite size should stay the same.
    initial_agent_size = AGENT_SPRITE.shape[0]
    new_size = rotated_agent.shape[0]
    agent_size_px = max(
        1, int(agent_radius_px * 2 * new_size / initial_agent_size)
    )
    # Making scale_size odd makes cropping math easier to center.
    if agent_size_px % 2 == 0:
        agent_size_px += 1
    resized_agent = cv2.resize(
        rotated_agent,
        (agent_size_px, agent_size_px),
        interpolation=cv2.INTER_LINEAR,
    )

    # The padding represents how much over the edge of the image the agent might
    # be, and corrects for that when pasting the agent into the main image.
    min_pad = (
        max(0, agent_size_px // 2 - relative_position[0]),
        max(0, agent_size_px // 2 - relative_position[1]),
    )

    max_pad = (
        max(
            0,
            (relative_position[0] + agent_size_px // 2 + 1)
            - im_position.shape[0],
        ),
        max(
            0,
            (relative_position[1] + agent_size_px // 2 + 1)
            - im_position.shape[1],
        ),
    )

    agent_patch = im_position[
        (relative_position[0] - agent_size_px // 2 + min_pad[0]) : (
            relative_position[0] + agent_size_px // 2 + 1 - max_pad[0]
        ),
        (relative_position[1] - agent_size_px // 2 + min_pad[1]) : (
            relative_position[1] + agent_size_px // 2 + 1 - max_pad[1]
        ),
        :,
    ]
    resized_agent = resized_agent[
        min_pad[0] : resized_agent.shape[0] - max_pad[0],
        min_pad[1] : resized_agent.shape[1] - max_pad[1],
    ]

    # Removes black background.
    agent_bg_mask = np.all(resized_agent > 10, axis=2)
    agent_patch[agent_bg_mask] = resized_agent[agent_bg_mask]

    # Rotate twice to fix coordinate system to upwards being positive-z.
    # Rotate instead of flip to keep agent rotations in sync with egocentric
    # view.
    im_position = np.rot90(im_position, 2)
    return im_position
