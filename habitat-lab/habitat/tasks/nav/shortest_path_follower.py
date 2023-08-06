#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import List, Optional, Union

import numpy as np

import habitat_sim
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
import magnum as mn
from habitat.utils.geometry_utils import quaternion_from_coeff, quaternion_rotate_vector
from habitat.tasks.utils import cartesian_to_polar

def action_to_one_hot(action: int) -> np.ndarray:
    one_hot = np.zeros(len(HabitatSimActions), dtype=np.float32)
    one_hot[action] = 1
    return one_hot


def _quat_to_xy_heading(quat):
    direction_vector = np.array([0, 0, -1])

    heading_vector = quaternion_rotate_vector(quat, direction_vector)

    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    return np.array([phi], dtype=np.float32)

class ShortestPathFollower:
    r"""Utility class for extracting the action on the shortest path to the
        goal.

    :param sim: HabitatSim instance.
    :param goal_radius: Distance between the agent and the goal for it to be
            considered successful.
    :param return_one_hot: If true, returns a one-hot encoding of the action
            (useful for training ML agents). If false, returns the
            SimulatorAction.
    :param stop_on_error: Return stop if the follower is unable to determine a
                          suitable action to take next.  If false, will raise
                          a habitat_sim.errors.GreedyFollowerError instead
    """

    def __init__(
        self,
        sim: HabitatSim,
        goal_radius: float,
        return_one_hot: bool = True,
        stop_on_error: bool = True,
    ):

        self._return_one_hot = return_one_hot
        self._sim = sim
        self._goal_radius = goal_radius
        self._follower: Optional[habitat_sim.GreedyGeodesicFollower] = None
        self._current_scene = None
        self._stop_on_error = stop_on_error

    def _build_follower(self):
        if self._current_scene != self._sim.habitat_config.scene:
            self._follower = self._sim.make_greedy_follower(
                0,
                self._goal_radius,
                stop_key=HabitatSimActions.stop,
                forward_key=HabitatSimActions.move_forward,
                left_key=HabitatSimActions.turn_left,
                right_key=HabitatSimActions.turn_right,
            )
            self._current_scene = self._sim.habitat_config.scene

    def _get_return_value(self, action) -> Union[int, np.ndarray]:
        if self._return_one_hot:
            return action_to_one_hot(action)
        else:
            return action

    def get_next_action(
        self, goal_pos: Union[List[float], np.ndarray]
    ) -> Optional[Union[int, np.ndarray]]:
        """Returns the next action along the shortest path."""
        self._build_follower()
        assert self._follower is not None
        try:
            curr_pos, curr_rot = None, None
            if isinstance(self._sim, RearrangeSim):
                ang_pos = float(self._sim.robot.base_rot) - np.pi / 2
                curr_quat = self._sim.robot.sim_obj.rotation
                curr_rotation = [
                    curr_quat.vector.x,
                    curr_quat.vector.y,
                    curr_quat.vector.z,
                    curr_quat.scalar,
                ]
                curr_quat = quaternion_from_coeff(
                   curr_rotation
                )
                # get heading angle
                rot = _quat_to_xy_heading(
                    curr_quat.inverse()
                )
                rot = rot - np.pi / 2
                # convert back to quaternion
                ang_pos = rot[0]
                curr_rot = mn.Quaternion(
                    mn.Vector3(0, np.sin(ang_pos / 2), 0), np.cos(ang_pos / 2)
                )
                curr_pos = self._sim.robot.base_pos

            # Get the target rotation
            next_action = self._follower.next_action_along(goal_pos, curr_rot=curr_rot, curr_pos=curr_pos)
        except habitat_sim.errors.GreedyFollowerError as e:
            if self._stop_on_error:
                next_action = HabitatSimActions.stop
            else:
                raise e

        return self._get_return_value(next_action)

    @property
    def mode(self):
        warnings.warn(".mode is depricated", DeprecationWarning)
        return ""

    @mode.setter
    def mode(self, new_mode: str):
        warnings.warn(".mode is depricated", DeprecationWarning)
