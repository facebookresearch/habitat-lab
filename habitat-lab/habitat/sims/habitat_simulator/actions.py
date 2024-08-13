#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""This module provides an extendable Enum singleton manager for mapping simulator action names to integer id values. Be default it provides cylinder agent move and look actions."""

from enum import Enum
from typing import Dict

import attr

from habitat.core.utils import Singleton


class _DefaultHabitatSimActions(Enum):
    """Enum class for default cylinder agent move and look actions. I.e. pointnav action space."""

    stop = 0
    move_forward = 1
    turn_left = 2
    turn_right = 3
    look_up = 4
    look_down = 5


@attr.s(auto_attribs=True, slots=True)
class HabitatSimActionsSingleton(metaclass=Singleton):
    """Implements an extendable Enum for the mapping of action names
    to their integer values.

    This means that new action names can be added, but old action names cannot
    be removed nor can their mapping be altered. This also ensures that all
    actions are always contiguously mapped in :py:`[0, len(HabitatSimActions) - 1]`

    This accessible as the global singleton :ref:`HabitatSimActions`
    """

    _known_actions: Dict[str, int] = attr.ib(init=False, factory=dict)

    def __attrs_post_init__(self):
        """Run after singleton initialization to register the default cylinder agent move and look actions."""
        for action in _DefaultHabitatSimActions:
            self._known_actions[action.name] = action.value

    def extend_action_space(self, name: str) -> int:
        """Extends the action space to accommodate a new action with
        the name :p:`name`

        :param name: The name of the new action
        :return: The number the action is registered on

        Usage:

        .. code:: py

            from habitat.sims.habitat_simulator.actions import HabitatSimActions
            HabitatSimActions.extend_action_space("MY_ACTION")
            print(HabitatSimActions.MY_ACTION)
        """
        assert (
            name not in self._known_actions
        ), "Cannot register an action name twice"
        self._known_actions[name] = len(self._known_actions)

        return self._known_actions[name]

    def has_action(self, name: str) -> bool:
        """Checks to see if action :p:`name` is already register

        :param name: The name to check
        :return: Whether or not :p:`name` already exists
        """

        return name in self._known_actions

    def __getattr__(self, name):
        return self._known_actions[name]

    def __getitem__(self, name):
        return self._known_actions[name]

    def __len__(self):
        return len(self._known_actions)

    def __iter__(self):
        return iter(self._known_actions)


HabitatSimActions: HabitatSimActionsSingleton = HabitatSimActionsSingleton()
