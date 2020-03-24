#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Optional, Union

import numpy as np

import habitat_sim
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim


def action_to_one_hot(action: int) -> np.array:
    one_hot = np.zeros(len(HabitatSimActions), dtype=np.float32)
    one_hot[action] = 1
    return one_hot


class ShortestPathFollower:
    r"""Utility class for extracting the action on the shortest path to the
        goal.
    Args:
        sim: HabitatSim instance.
        goal_radius: Distance between the agent and the goal for it to be
            considered successful.
        return_one_hot: If true, returns a one-hot encoding of the action
            (useful for training ML agents). If false, returns the
            SimulatorAction.
    """

    def __init__(
        self, sim: HabitatSim, goal_radius: float, return_one_hot: bool = True
    ):
        self._follower = habitat_sim.nav.GreedyGeodesicFollower(
            sim._sim.pathfinder,
            sim._sim.get_agent(0),
            goal_radius=goal_radius,
            stop_key=HabitatSimActions.STOP,
            forward_key=HabitatSimActions.MOVE_FORWARD,
            left_key=HabitatSimActions.TURN_LEFT,
            right_key=HabitatSimActions.TURN_RIGHT,
        )
        self._return_one_hot = return_one_hot

    def _get_return_value(self, action) -> Union[int, np.array]:
        if self._return_one_hot:
            return action_to_one_hot(action)
        else:
            return action

    def get_next_action(
        self, goal_pos: np.array
    ) -> Optional[Union[int, np.array]]:
        """Returns the next action along the shortest path.
        """
        next_action = self._follower.next_action_along(goal_pos)

        return self._get_return_value(next_action)

    @property
    def mode(self):
        warnings.warn(".mode is depricated", DeprecationWarning)
        return ""

    @mode.setter
    def mode(self, new_mode: str):
        warnings.warn(".mode is depricated", DeprecationWarning)
