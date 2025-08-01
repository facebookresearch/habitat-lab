#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import magnum as mn
import numpy as np

from habitat_sim.gfx import DebugLineRender

import roboticstoolbox as rtb
from scipy.spatial.transform import Rotation as R


# --- Load robot ---
robot = rtb.models.Panda()

# --- Joint limits ---
joint_mins = np.array([-2.7437, -1.7837, -2.9007, -2.9, -2.8065, 0.5445, -3.0159])
joint_maxs = np.array([ 2.7437,  1.7837,  2.9007, -0.1518, 2.8065, 4.5169, 3.0159])


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



# --- FK function: returns RPY in degrees ---
def get_ee_rpy_deg(q):
    T = robot.fkine(q, end=robot.links[8])
    R_ee = T.R
    return R.from_matrix(R_ee).as_euler('xyz', degrees=True)

# --- RPY distance ---
def rpy_distance_deg(rpy1, rpy2):
    """ Calculate the distance between two RPY angles in degrees.
    The distance is calculated as the shortest angle distance between the two angles.
    """
    diff = np.abs(rpy1 - rpy2)
    diff = (diff + 180) % 360 - 180  # shortest angle distance
    return np.linalg.norm(diff)


def get_ee_position(q):
    T = robot.fkine(q, end=robot.links[8])
    return T.t

def get_ee_pose(q):
    T = robot.fkine(q, end=robot.links[8])
    return T


def get_basis_vectors_lengths(positions):
    """ Given a set of 3D positions (Nx3 numpy array), return basis vectors along each axis
        separated into positive and negative halves. Each returned array is 1D.
    """
    if len(positions) == 0:
        return np.array([0]*6)

    positions = np.asarray(positions)

    x_plus = []
    y_plus = []
    z_plus = []
    x_minus = []
    y_minus = []
    z_minus = []

    normalization_factors = [0,0,0,0,0,0] # [x+, x-, y+, y-, z+, z-]
    norm_threshold = 0.07  # Threshold to ignore small vectors
    position_threshold = 0.05

    for position in positions:
        norm = np.linalg.norm(position)
        if norm < norm_threshold:
            continue

        
        if position[0] > 0:
            x_plus.append(position[0]*norm if abs(position[0]) > position_threshold else 0.0)
        else:
            normalization_factors[3] = normalization_factors[3] + norm

        if position[1] > 0:
            y_plus.append(position[1]*norm if abs(position[1]) > position_threshold else 0.0)
        else:
            normalization_factors[4] = normalization_factors[4] + norm

        if position[2] > 0:
            z_plus.append(position[2]*norm if abs(position[2]) > position_threshold else 0.0)
        else:
            normalization_factors[5] = normalization_factors[5] + norm

        if position[0] < 0:
            x_minus.append(-position[0]*norm if abs(position[0]) > position_threshold else 0.0)
        else:
            normalization_factors[0] = normalization_factors[0] + norm

        if position[1] < 0:
            y_minus.append(-position[1]*norm if abs(position[1]) > position_threshold else 0.0)
        else:
            normalization_factors[1] = normalization_factors[1] + norm

        if position[2] < 0:
            z_minus.append(-position[2]*norm if abs(position[2]) > position_threshold else 0.0)
        else:
            normalization_factors[2] = normalization_factors[2] + norm


    if len(x_plus) <= 1: x_plus, normalization_factors[0] = [0.0], 1.0
    if len(y_plus) <= 1: y_plus, normalization_factors[1] = [0.0], 1.0
    if len(z_plus) <= 1: z_plus, normalization_factors[2] = [0.0], 1.0
    if len(x_minus) <= 1: x_minus, normalization_factors[3] = [0.0], 1.0
    if len(y_minus) <= 1: y_minus, normalization_factors[4] = [0.0], 1.0
    if len(z_minus) <= 1: z_minus, normalization_factors[5] = [0.0], 1.0


    summary = np.array([
        np.sum(x_plus)/normalization_factors[0], np.sum(y_plus)/normalization_factors[1],
        np.sum(z_plus)/normalization_factors[2], np.sum(x_minus)/normalization_factors[3],
        np.sum(y_minus)/normalization_factors[4], np.sum(z_minus)/normalization_factors[5]
    ])

    return summary
