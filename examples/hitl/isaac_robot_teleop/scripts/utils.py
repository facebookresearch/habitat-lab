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
