#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# GuiInput relies on the magnum.platform.glfw.Application.KeyEvent.Key enum and similar for mouse buttons. On headless systems, we may be unable to import magnum.platform.glfw.Application. Fall back to a stub implementation of GuiInput in that case.
do_agnostic_gui_input = False
try:
    from magnum.platform.glfw import Application
except ImportError:
    print(
        "GuiInput warning: Failed to magnum.platform.glfw. Falling back to agnostic implementation for use with headless server. Local keyboard/mouse input won't work."
    )
    do_agnostic_gui_input = True

if do_agnostic_gui_input:
    from enum import Enum

    # physical key enum from USB HID Usage Tables
    # https://www.usb.org/sites/default/files/documents/hut1_12v2.pdf, page 53
    # add missing keys here as necessary
    class AgnosticKeyNS(Enum):
        A = 0x04
        B = 0x05
        C = 0x06
        D = 0x07
        E = 0x08
        F = 0x09
        G = 0x0A
        H = 0x0B
        I = 0x0C
        J = 0x0D
        K = 0x0E
        L = 0x0F
        M = 0x10
        N = 0x11
        O = 0x12
        P = 0x13
        Q = 0x14
        R = 0x15
        S = 0x16
        T = 0x17
        U = 0x18
        V = 0x19
        W = 0x1A
        X = 0x1B
        Y = 0x1C
        Z = 0x1D
        ZERO = 0x27
        ONE = 0x1E
        TWO = 0x1F
        THREE = 0x20
        FOUR = 0x21
        FIVE = 0x22
        SIX = 0x23
        SEVEN = 0x24
        EIGHT = 0x25
        NINE = 0x26
        ESC = 0x29
        SPACE = 0x2C
        TAB = 0x2B

    class StubNSMeta(type):
        def __getattr__(cls, name):
            return None

    # Stub version of Application.MouseEvent.Button
    class StubMouseNS(metaclass=StubNSMeta):
        pass


class GuiInput:
    """
    Container to hold the state of keyboard/mouse input.

    This class isn't usable by itself for getting input from the underlying OS. I.e. it won't self-populate from underlying OS input APIs. See also gui_application.py InputHandlerApplication.
    """

    if do_agnostic_gui_input:
        KeyNS = AgnosticKeyNS
        MouseNS = StubMouseNS
    else:
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

    @property
    def is_stub_implementation(self):
        """
        Indicates whether this is a stub implementation. If so, it'll return False for all queries like get_key_down(...).
        """
        return do_agnostic_gui_input

    def validate_key(key):
        if not do_agnostic_gui_input:
            assert isinstance(key, Application.KeyEvent.Key)

    def get_key(self, key):
        GuiInput.validate_key(key)
        return key in self._key_held

    def get_any_key_down(self):
        return len(self._key_down) > 0

    def get_key_down(self, key):
        GuiInput.validate_key(key)
        return key in self._key_down

    def get_key_up(self, key):
        GuiInput.validate_key(key)
        return key in self._key_up

    def validate_mouse_button(mouse_button):
        if not do_agnostic_gui_input:
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
