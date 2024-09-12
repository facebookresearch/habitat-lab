#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from habitat_hitl.core.key_mapping import KeyCode, MouseButton

if TYPE_CHECKING:
    from habitat_sim.geo import Ray


class GuiInput:
    """
    Container to hold the state of keyboard/mouse input.

    This class isn't usable by itself for getting input from the underlying OS. I.e. it won't self-populate from underlying OS input APIs. See also gui_application.py InputHandlerApplication.
    """

    def __init__(self):
        self._key_held = set()
        self._mouse_button_held = set()
        self._mouse_position = [0, 0]

        self._key_down = set()
        self._key_up = set()
        self._mouse_button_down = set()
        self._mouse_button_up = set()
        self._relative_mouse_position = [0, 0]
        self._mouse_scroll_offset = 0.0
        self._mouse_ray: Optional[Ray] = None

    def validate_key(key):
        assert isinstance(key, KeyCode)

    def get_key(self, key):
        GuiInput.validate_key(key)
        return key in self._key_held

    def get_any_key_down(self):
        return len(self._key_down) > 0

    def get_any_input(self) -> bool:
        """Returns true if any input is active."""
        return (
            len(self._key_down) > 0
            or len(self._key_up) > 0
            or len(self._mouse_button_down) > 0
            or len(self._mouse_button_up) > 0
        )

    def get_key_down(self, key):
        GuiInput.validate_key(key)
        return key in self._key_down

    def get_key_up(self, key):
        GuiInput.validate_key(key)
        return key in self._key_up

    def validate_mouse_button(mouse_button):
        assert isinstance(mouse_button, MouseButton)

    def get_mouse_button(self, mouse_button):
        GuiInput.validate_mouse_button(mouse_button)
        return mouse_button in self._mouse_button_held

    def get_mouse_button_down(self, mouse_button):
        GuiInput.validate_mouse_button(mouse_button)
        return mouse_button in self._mouse_button_down

    def get_mouse_button_up(self, mouse_button):
        GuiInput.validate_mouse_button(mouse_button)
        return mouse_button in self._mouse_button_up

    @property
    def mouse_position(self):
        return self._mouse_position

    @property
    def relative_mouse_position(self):
        return self._relative_mouse_position

    @property
    def mouse_scroll_offset(self):
        return self._mouse_scroll_offset

    @property
    def mouse_ray(self):
        return self._mouse_ray

    def reset(self, reset_continuous_input: bool = True):
        """
        Reset the input states. To be called at the end of a frame.

        `reset_continuous_input`: controls whether to reset continuous input like scrolling or dragging.
        Remote clients send their input at a different frequency than the server framerate.
        To avoid choppiness, their continuous inputs should be reset before consolidating new remote inputs.
        This differs from discrete input like clicking, which must be reset every frame to avoid extending click events across multiple frames.
        """
        self._key_down.clear()
        self._key_up.clear()
        self._mouse_button_down.clear()
        self._mouse_button_up.clear()

        if reset_continuous_input:
            self._relative_mouse_position = [0, 0]
            self._mouse_scroll_offset = 0.0

    def copy_from(self, other: GuiInput):
        self._key_down = set(other._key_down)
        self._key_up = set(other._key_up)
        self._key_held = set(other._key_held)
        self._mouse_button_down = set(other._mouse_button_down)
        self._mouse_button_up = set(other._mouse_button_up)
        self._mouse_button_held = set(other._mouse_button_held)
        self._mouse_position = list(other._mouse_position)
        self._relative_mouse_position = list(other._relative_mouse_position)
        self._mouse_scroll_offset = other._mouse_scroll_offset
        self._mouse_ray = other._mouse_ray
