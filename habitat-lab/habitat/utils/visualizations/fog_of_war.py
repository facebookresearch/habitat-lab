#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numba
import numpy as np

from habitat.utils.visualizations import maps


@numba.jit(nopython=True)
def bresenham_supercover_line(pt1, pt2):
    r"""Line drawing algo based
    on http://eugen.dedu.free.fr/projects/bresenham/
    """

    ystep, xstep = 1, 1

    x, y = pt1
    dx, dy = pt2 - pt1

    if dy < 0:
        ystep *= -1
        dy *= -1

    if dx < 0:
        xstep *= -1
        dx *= -1

    line_pts = [[x, y]]

    ddx, ddy = 2 * dx, 2 * dy
    if ddx > ddy:
        errorprev = dx
        error = dx
        for _ in range(int(dx)):
            x += xstep
            error += ddy

            if error > ddx:
                y += ystep
                error -= ddx
                if error + errorprev < ddx:
                    line_pts.append([x, y - ystep])
                elif error + errorprev > ddx:
                    line_pts.append([x - xstep, y])
                else:
                    line_pts.append([x - xstep, y])
                    line_pts.append([x, y - ystep])

            line_pts.append([x, y])

            errorprev = error
    else:
        errorprev = dx
        error = dx
        for _ in range(int(dy)):
            y += ystep
            error += ddx

            if error > ddy:
                x += xstep
                error -= ddy
                if error + errorprev < ddy:
                    line_pts.append([x - xstep, y])
                elif error + errorprev > ddy:
                    line_pts.append([x, y - ystep])
                else:
                    line_pts.append([x - xstep, y])
                    line_pts.append([x, y - ystep])

            line_pts.append([x, y])

            errorprev = error

    return line_pts


@numba.jit(nopython=True)
def draw_fog_of_war_line(top_down_map, fog_of_war_mask, pt1, pt2):
    r"""Draws a line on the fog_of_war_mask mask between pt1 and pt2"""

    for pt in bresenham_supercover_line(pt1, pt2):
        x, y = pt

        if x < 0 or x >= fog_of_war_mask.shape[0]:
            break

        if y < 0 or y >= fog_of_war_mask.shape[1]:
            break

        if top_down_map[x, y] == maps.MAP_INVALID_POINT:
            break

        fog_of_war_mask[x, y] = 1


@numba.jit(nopython=True)
def _draw_loop(
    top_down_map,
    fog_of_war_mask,
    current_point,
    current_angle,
    max_line_len,
    angles,
):
    for angle in angles:
        draw_fog_of_war_line(
            top_down_map,
            fog_of_war_mask,
            current_point,
            current_point
            + max_line_len
            * np.array(
                [np.cos(current_angle + angle), np.sin(current_angle + angle)]
            ),
        )


def reveal_fog_of_war(
    top_down_map: np.ndarray,
    current_fog_of_war_mask: np.ndarray,
    current_point: np.ndarray,
    current_angle: float,
    fov: float = 90,
    max_line_len: float = 100,
) -> np.ndarray:
    r"""Reveals the fog-of-war at the current location

    This works by simply drawing lines from the agents current location
    and stopping once a wall is hit

    Args:
        top_down_map: The current top down map.  Used for respecting walls when revealing
        current_fog_of_war_mask: The current fog-of-war mask to reveal the fog-of-war on
        current_point: The current location of the agent on the fog_of_war_mask
        current_angle: The current look direction of the agent on the fog_of_war_mask
        fov: The feild of view of the agent
        max_line_len: The maximum length of the lines used to reveal the fog-of-war

    Returns:
        The updated fog_of_war_mask
    """
    fov = np.deg2rad(fov)

    # Set the angle step to a value such that delta_angle * max_line_len = 1
    angles = np.arange(
        -fov / 2, fov / 2, step=1.0 / max_line_len, dtype=np.float32
    )

    fog_of_war_mask = current_fog_of_war_mask.copy()
    _draw_loop(
        top_down_map,
        fog_of_war_mask,
        current_point,
        current_angle,
        max_line_len,
        angles,
    )

    return fog_of_war_mask
