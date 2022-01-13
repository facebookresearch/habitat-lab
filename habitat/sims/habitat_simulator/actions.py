#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Dict

import attr

import habitat_sim
from habitat.core.registry import registry
from habitat.core.simulator import ActionSpaceConfiguration
from habitat.core.utils import Singleton


class _DefaultHabitatSimActions(Enum):
    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    LOOK_UP = 4
    LOOK_DOWN = 5


@attr.s(auto_attribs=True, slots=True)
class HabitatSimActionsSingleton(metaclass=Singleton):
    r"""Implements an extendable Enum for the mapping of action names
    to their integer values.

    This means that new action names can be added, but old action names cannot
    be removed nor can their mapping be altered. This also ensures that all
    actions are always contigously mapped in :py:`[0, len(HabitatSimActions) - 1]`

    This accesible as the global singleton :ref:`HabitatSimActions`
    """

    _known_actions: Dict[str, int] = attr.ib(init=False, factory=dict)

    def __attrs_post_init__(self):
        for action in _DefaultHabitatSimActions:
            self._known_actions[action.name] = action.value

    def extend_action_space(self, name: str) -> int:
        r"""Extends the action space to accommodate a new action with
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
        r"""Checks to see if action :p:`name` is already register

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


@registry.register_action_space_configuration(name="v0")
class HabitatSimV0ActionSpaceConfiguration(ActionSpaceConfiguration):
    def get(self):
        return {
            HabitatSimActions.STOP: habitat_sim.ActionSpec("stop"),
            HabitatSimActions.MOVE_FORWARD: habitat_sim.ActionSpec(
                "move_forward",
                habitat_sim.ActuationSpec(
                    amount=self.config.FORWARD_STEP_SIZE
                ),
            ),
            HabitatSimActions.TURN_LEFT: habitat_sim.ActionSpec(
                "turn_left",
                habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE),
            ),
            HabitatSimActions.TURN_RIGHT: habitat_sim.ActionSpec(
                "turn_right",
                habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE),
            ),
        }


@registry.register_action_space_configuration(name="v1")
class HabitatSimV1ActionSpaceConfiguration(
    HabitatSimV0ActionSpaceConfiguration
):
    def get(self):
        config = super().get()
        new_config = {
            HabitatSimActions.LOOK_UP: habitat_sim.ActionSpec(
                "look_up",
                habitat_sim.ActuationSpec(amount=self.config.TILT_ANGLE),
            ),
            HabitatSimActions.LOOK_DOWN: habitat_sim.ActionSpec(
                "look_down",
                habitat_sim.ActuationSpec(amount=self.config.TILT_ANGLE),
            ),
        }

        config.update(new_config)

        return config


@registry.register_action_space_configuration(name="pyrobotnoisy")
class HabitatSimPyRobotActionSpaceConfiguration(ActionSpaceConfiguration):
    def get(self):
        return {
            HabitatSimActions.STOP: habitat_sim.ActionSpec("stop"),
            HabitatSimActions.MOVE_FORWARD: habitat_sim.ActionSpec(
                "pyrobot_noisy_move_forward",
                habitat_sim.PyRobotNoisyActuationSpec(
                    amount=self.config.FORWARD_STEP_SIZE,
                    robot=self.config.NOISE_MODEL.ROBOT,
                    controller=self.config.NOISE_MODEL.CONTROLLER,
                    noise_multiplier=self.config.NOISE_MODEL.NOISE_MULTIPLIER,
                ),
            ),
            HabitatSimActions.TURN_LEFT: habitat_sim.ActionSpec(
                "pyrobot_noisy_turn_left",
                habitat_sim.PyRobotNoisyActuationSpec(
                    amount=self.config.TURN_ANGLE,
                    robot=self.config.NOISE_MODEL.ROBOT,
                    controller=self.config.NOISE_MODEL.CONTROLLER,
                    noise_multiplier=self.config.NOISE_MODEL.NOISE_MULTIPLIER,
                ),
            ),
            HabitatSimActions.TURN_RIGHT: habitat_sim.ActionSpec(
                "pyrobot_noisy_turn_right",
                habitat_sim.PyRobotNoisyActuationSpec(
                    amount=self.config.TURN_ANGLE,
                    robot=self.config.NOISE_MODEL.ROBOT,
                    controller=self.config.NOISE_MODEL.CONTROLLER,
                    noise_multiplier=self.config.NOISE_MODEL.NOISE_MULTIPLIER,
                ),
            ),
            HabitatSimActions.LOOK_UP: habitat_sim.ActionSpec(
                "look_up",
                habitat_sim.ActuationSpec(amount=self.config.TILT_ANGLE),
            ),
            HabitatSimActions.LOOK_DOWN: habitat_sim.ActionSpec(
                "look_down",
                habitat_sim.ActuationSpec(amount=self.config.TILT_ANGLE),
            ),
            # The perfect actions are needed for the oracle planner
            "_forward": habitat_sim.ActionSpec(
                "move_forward",
                habitat_sim.ActuationSpec(
                    amount=self.config.FORWARD_STEP_SIZE
                ),
            ),
            "_left": habitat_sim.ActionSpec(
                "turn_left",
                habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE),
            ),
            "_right": habitat_sim.ActionSpec(
                "turn_right",
                habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE),
            ),
        }


@registry.register_action_space_configuration(name="velocitycontrol")
class HabitatSimVelocityCtrlActionSpaceConfiguration(ActionSpaceConfiguration):
    def get(self):
        return {
            HabitatSimActions.VELOCITY_CTRL: habitat_sim.ActionSpec(
                "velocity_control"
            ),
        }
