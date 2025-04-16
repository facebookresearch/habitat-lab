#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Final

from habitat_hitl.core.key_mapping import XRButton

NUM_CONTROLLERS: Final[int] = 2
HAND_LEFT: Final[int] = 0
HAND_RIGHT: Final[int] = 1


class XRController:
    """
    State of a single XR controller.
    """

    def __init__(self):
        self._buttons_held: set[XRButton] = set()
        self._buttons_down: set[XRButton] = set()
        self._buttons_up: set[XRButton] = set()
        self._buttons_touched: set[XRButton] = set()
        self._thumbstick_axis: list[float] = [0.0, 0.0]
        self._hand_trigger: float = 0.0
        self._index_trigger: float = 0.0
        self._is_controller_in_hand: bool = False

    def validate_button(button):
        assert isinstance(button, XRButton)

    def get_button(self, button):
        XRController.validate_button(button)
        return button in self._buttons_held

    def get_button_down(self, button):
        XRController.validate_button(button)
        return button in self._buttons_down

    def get_button_up(self, button):
        XRController.validate_button(button)
        return button in self._buttons_up

    def get_button_touched(self, button):
        XRController.validate_button(button)
        return button in self._buttons_touched

    def get_thumbstick(self):
        return self._thumbstick_axis

    def get_index_trigger(self):
        return self._index_trigger

    def get_hand_trigger(self):
        return self._hand_trigger

    def get_is_controller_in_hand(self):
        return self._is_controller_in_hand

    def reset(self, reset_continuous_input: bool = True):
        self._buttons_down.clear()
        self._buttons_up.clear()

        if reset_continuous_input:
            self._thumbstick_axis = [0, 0]
            self._hand_trigger = 0.0
            self._index_trigger = 0.0


class XRInput:
    """
    State of the XR input system (HMD and controllers).
    """

    def __init__(self):
        self._controllers: list[XRController] = []
        for _ in range(NUM_CONTROLLERS):
            self._controllers.append(XRController())

        self._origin_position: list[float] = [0.0, 0.0, 0.0]
        self._origin_rotation: list[float] = [0.0, 0.0, 0.0, 0.0]

    @property
    def controllers(self):
        return self._controllers

    @property
    def left_controller(self):
        return self._controllers[HAND_LEFT]

    @property
    def right_controller(self):
        return self._controllers[HAND_RIGHT]

    @property
    def origin_position(self):
        return self._origin_position

    @property
    def origin_rotation(self):
        return self._origin_rotation

    def reset(self, reset_continuous_input: bool = True):
        """
        Reset the input states. To be called at the end of a frame.

        `reset_continuous_input`: controls whether to reset continuous input like scrolling or dragging.
        Remote clients send their input at a different frequency than the server framerate.
        To avoid choppiness, their continuous inputs should be reset before consolidating new remote inputs.
        This differs from discrete input like clicking, which must be reset every frame to avoid extending click events across multiple frames.
        """
        for controller in self._controllers:
            controller.reset(reset_continuous_input)
