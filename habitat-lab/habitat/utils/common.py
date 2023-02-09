#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from os import makedirs
from os import path as osp
from typing import List

from habitat.core.logging import logger


def check_make_dir(directory_path: str) -> bool:
    """
    Check for the existence of the provided directory_path and create it if not found.
    """
    # if output directory doesn't exist, create it
    if not osp.exists(directory_path):
        try:
            makedirs(directory_path)
        except OSError:
            logger.error(
                f"check_make_dir: Failed to create the specified directory_path: {directory_path}"
            )
            return False
        logger.info(
            f"check_make_dir: directory_path did not exist and was created: {directory_path}"
        )
    return True


def cull_string_list_by_substrings(
    full_list: List[str],
    included_substrings: List[str],
    excluded_substrings: List[str],
) -> List[str]:
    """
    Cull a list of strings to the subset of strings containing any of the "included_substrings" and none of the "excluded_substrings".
    Returns the culled list, does not modify the input list.
    """
    culled_list: List[str] = []
    for string in full_list:
        excluded = False
        for excluded_substring in excluded_substrings:
            if excluded_substring in string:
                excluded = True
                break
        if not excluded:
            for included_substring in included_substrings:
                if included_substring in string:
                    culled_list.append(string)
                    break
    return culled_list





def transform_global_to_base(XYT, current_pose, trans=None):
    """
    Transforms the point cloud into geocentric frame to account for
    camera position
    """
    goal_pos = trans.inverted().transform_point(
        np.array([XYT[0], 0.0, XYT[1]])
    )
    error_t = XYT[2] - current_pose[2]
    error_t = (error_t + np.pi) % (2.0*np.pi) - np.pi
    error_x = goal_pos[0]
    error_y = goal_pos[2]
    return [error_x, error_y, error_t]




class continous_controller:
    """A controller that transforms waypoints to velocity command"""
    def __init__(self, v_max=1.0, w_max=1.0, acc_lin=2.4, acc_ang=2.4, \
        max_heading_ang=np.pi / 10, \
        lin_error_tol=0.001, ang_error_tol=0.001,\
        track_yaw=True):
        self.track_yaw = track_yaw

        # Params
        self.v_max = v_max
        self.w_max = w_max

        self.acc_lin = acc_lin
        self.acc_ang = acc_ang

        self.max_heading_ang = max_heading_ang

        self.lin_error_tol = lin_error_tol
        self.ang_error_tol = ang_error_tol

        # Init
        self.xyt_goal = np.zeros(3)
        self.dxyt_goal = np.zeros(3)

    def set_goal(self, goal, vel_goal=None):
        self.xyt_goal = goal
        if vel_goal is not None:
            self.dxyt_goal = vel_goal

    def _compute_error_pose(self, xyt_base, trans=None):
        """
        Updates error based on robot localization
        """
        xyt_err = transform_global_to_base(self.xyt_goal, xyt_base, trans)
        if not self.track_yaw:
            xyt_err[2] = 0.0

        return xyt_err

    @staticmethod
    def _velocity_feedback_control(x_err, a, v_max):
        """
        Computes velocity based on distance from target.
        Used for both linear and angular motion.

        Current implementation: Trapezoidal velocity profile
        """
        t = np.sqrt(2.0 * abs(x_err) / a)
        v = min(a * t, v_max)
        return v * np.sign(x_err)

    @staticmethod
    def _turn_rate_limit(lin_err, heading_diff, w_max, max_heading_ang, tol=0.0):
        """
        Compute velocity limit that prevents path from overshooting goal

        heading error decrease rate > linear error decrease rate
        (w - v * np.sin(phi) / D) / phi > v * np.cos(phi) / D
        v < (w / phi) / (np.sin(phi) / D / phi + np.cos(phi) / D)
        v < w * D / (np.sin(phi) + phi * np.cos(phi))

        (D = linear error, phi = angular error)
        """
        assert lin_err >= 0.0
        assert heading_diff >= 0.0
        if heading_diff > max_heading_ang:
            return 0.0
        else:
            return (
                w_max
                * lin_err
                / (
                    np.sin(heading_diff)
                    + heading_diff * np.cos(heading_diff)
                    + 1e-5
                )
            )

    def _feedback_traj_track(self, xyt_err):
        xyt_err = self._compute_error_pose(xyt)
        v_raw = (
            V_MAX_DEFAULT
            * (K1 * xyt_err[0] + xyt_err[1] * np.tan(xyt_err[2]))
            / np.cos(xyt_err[2])
        )
        w_raw = (
            V_MAX_DEFAULT
            * (K2 * xyt_err[1] + K3 * np.tan(xyt_err[2]))
            / np.cos(xyt_err[2]) ** 2
        )
        v_out = min(v_raw, V_MAX_DEFAULT)
        w_out = min(w_raw, W_MAX_DEFAULT)
        return np.array([v_out, w_out])

    def _feedback_simple(self, xyt_err):
        v_cmd = w_cmd = 0

        lin_err_abs = np.linalg.norm(xyt_err[0:2])
        ang_err = xyt_err[2]

        # Go to goal XY position if not there yet
        if lin_err_abs > self.lin_error_tol:
            heading_err = np.arctan2(xyt_err[1], xyt_err[0])
            heading_err_abs = abs(heading_err)
            # Compute linear velocity
            v_raw = self._velocity_feedback_control(
                lin_err_abs, self.acc_lin, self.v_max
            )
            v_limit = self._turn_rate_limit(
                lin_err_abs,
                heading_err_abs,
                self.w_max / 2.0,
                self.max_heading_ang,
                tol=self.lin_error_tol,
            )

            v_cmd = np.clip(v_raw, 0.0, v_limit)

            # Compute angular velocity
            w_cmd = self._velocity_feedback_control(
                heading_err, self.acc_ang, self.w_max
            )

        # Rotate to correct yaw if yaw tracking is on and XY position is at goal
        elif abs(ang_err) > self.ang_error_tol and self.track_yaw:
            # Compute angular velocity
            w_cmd = self._velocity_feedback_control(
                ang_err, self.acc_ang, self.w_max
            )
        return v_cmd, w_cmd

    def forward(self, xyt, trans):
        xyt_err = self._compute_error_pose(xyt, trans)
        return self._feedback_simple(xyt_err)
