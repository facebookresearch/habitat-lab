#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import magnum as mn
import numpy as np

from habitat_sim.gfx import DebugLineRender


def LERP(vec0: List[float], vec1: List[float], t: float) -> List[float]:
    """
    Linear Interpolation (LERP) for two vectors (lists of floats) representing, for example, a joint space pose.
    Requires len(vec0) == len(vec1)
    """
    if len(vec0) != len(vec1):
        print(f"Cannot LERP mismatching vectors {len(vec0)} vs {len(vec1)}")
    npv0 = np.array(vec0)
    npv1 = np.array(vec1)
    delta = npv1 - npv0
    return list(npv0 + delta * t)


def debug_draw_axis(
    dblr: DebugLineRender, transform: mn.Matrix4 = None, scale: float = 1.0
) -> None:
    if transform is not None:
        dblr.push_transform(transform)
    for unit_axis in range(3):
        vec = mn.Vector3()
        vec[unit_axis] = 1.0
        color = mn.Color3(0.5)
        color[unit_axis] = 1.0
        dblr.draw_transformed_line(mn.Vector3(), vec * scale, color)
    if transform is not None:
        dblr.pop_transform()


def normalize_angle(angle: float) -> float:
    """
    normalize an angle into the range [-pi, pi]
    """
    mod_angle = mn.math.fmod(angle, 2 * mn.math.pi)
    if mod_angle > mn.math.pi:
        mod_angle -= mn.math.pi * 2
    elif mod_angle <= -mn.math.pi:
        mod_angle += mn.math.pi * 2
    return mod_angle


def project_point(
    camera_local_to_global: np.ndarray,
    point_3d: np.ndarray,
    hfov=80,
    near=0.01,
    far=50,
    resolution=(600, 960),
):
    # Get camera's position and orientation from local to global transform
    camera_global_to_local = np.linalg.inv(camera_local_to_global)

    # Transform point to camera's local space
    point_cam = np.dot(camera_global_to_local, np.append(point_3d, 1))[:3]

    # Compute projection matrix
    aspect_ratio = resolution[1] / resolution[0]  # width / height
    # print(f"aspect_ratio = {aspect_ratio}")
    f = 1 / np.tan(np.radians(hfov) / 2)
    proj_mat = np.array(
        [
            [f / aspect_ratio, 0, 0, 0],
            [0, f, 0, 0],
            [
                0,
                0,
                -(far + near) / (far - near),
                -2 * far * near / (far - near),
            ],
            [0, 0, -1, 0],
        ]
    )
    # print(f"proj_mat = {proj_mat}")

    # Project point to clip space
    point_clip = np.dot(proj_mat, np.append(point_cam, 1))
    if point_clip[3] != 0:
        point_clip = point_clip / point_clip[3]  # Perspective divide

    # Convert to screen space ([-1,1] to [0, width] and [0, height])
    x_screen = (point_clip[0] * 0.5 + 0.5) * resolution[1]
    y_screen = (1 - (point_clip[1] * 0.5 + 0.5)) * resolution[0]  # Flip y

    return x_screen, y_screen, point_clip


def is_in_frustum(point_clip, bound=0.61):
    """
    NOTE: bound is empirically chosen base on local testing to match the baked camera intrinsics.
    """
    # Check if point is inside clip space bounds (-1 to 1 in x, y; -1 to 1 in z considering perspective)
    return (
        -bound <= point_clip[0] <= bound and -bound <= point_clip[1] <= bound
    )


def point_in_frustum(camera_local_to_global: np.ndarray, point_3d: np.ndarray):
    _screen_x, _screen_y, point_projected = project_point(
        camera_local_to_global, point_3d
    )
    # print(f"Projected to {point_projected}, ({screen_x}, {screen_y})")
    return is_in_frustum(point_projected)
