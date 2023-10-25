#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from magnum.platform.glfw import Application


# This key and mouse-button is API based loosely on https://docs.unity3d.com/ScriptReference/Input.html
class GuiInput:
    KeyNS = Application.KeyEvent.Key
    MouseNS = Application.MouseEvent.Button

    def __init__(self):
        self._key_held = set()
        self._mouse_button_held = set()
        self._mouse_position = [0, 0]

        self._key_down = set()
        self._key_up = set()
        self._mouse_button_down = set()
        self._mouse_button_up = set()
        self._relative_mouse_position = [0, 0]
        self._mouse_scroll_offset = 0
        self._mouse_ray = None

    def shallow_copy_from(self, other):
        self._key_held = other._key_held
        self._mouse_button_held = other._mouse_button_held
        self._mouse_position = other._mouse_position

        self._key_down = other._key_down
        self._key_up = other._key_up
        self._mouse_button_down = other._mouse_button_down
        self._mouse_button_up = other._mouse_button_up
        self._relative_mouse_position = other._relative_mouse_position
        self._mouse_scroll_offset = other._mouse_scroll_offset
        self._mouse_ray = other._mouse_ray

    def validate_key(key):
        assert isinstance(key, Application.KeyEvent.Key)

    def get_key(self, key):
        GuiInput.validate_key(key)
        return key in self._key_held

    def get_key_down(self, key):
        GuiInput.validate_key(key)
        return key in self._key_down

    def get_key_up(self, key):
        GuiInput.validate_key(key)
        return key in self._key_up

    def validate_mouse_button(mouse_button):
        assert isinstance(mouse_button, Application.MouseEvent.Button)

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

    # Key/button up/down is only True on the frame it occurred. Mouse relative position is
    # relative to its position at the start of frame.
    def on_frame_end(self):
        self._key_down.clear()
        self._key_up.clear()
        self._mouse_button_down.clear()
        self._mouse_button_up.clear()
        self._relative_mouse_position = [0, 0]
        self._mouse_scroll_offset = 0
