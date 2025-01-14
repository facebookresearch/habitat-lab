#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import EnumMeta, IntEnum
from typing import Any, Dict, Optional, Set


class KeyCodeMetaEnum(EnumMeta):
    keycode_value_cache: Set[int] = None

    # Override 'in' keyword to check whether the specified integer exists in 'KeyCode'.
    def __contains__(cls, value) -> bool:
        if KeyCodeMetaEnum.keycode_value_cache == None:
            KeyCodeMetaEnum.keycode_value_cache = set(KeyCode)
        return value in KeyCodeMetaEnum.keycode_value_cache


class KeyCode(IntEnum, metaclass=KeyCodeMetaEnum):
    """
    Input keys available to control habitat-hitl.
    """

    # Physical key enum from USB HID Usage Tables
    # https://www.usb.org/sites/default/files/documents/hut1_12v2.pdf, page 53

    # fmt: off
    A       = 0x04
    B       = 0x05
    C       = 0x06
    D       = 0x07
    E       = 0x08
    F       = 0x09
    G       = 0x0A
    H       = 0x0B
    I       = 0x0C
    J       = 0x0D
    K       = 0x0E
    L       = 0x0F
    M       = 0x10
    N       = 0x11
    O       = 0x12
    P       = 0x13
    Q       = 0x14
    R       = 0x15
    S       = 0x16
    T       = 0x17
    U       = 0x18
    V       = 0x19
    W       = 0x1A
    X       = 0x1B
    Y       = 0x1C
    Z       = 0x1D
    ZERO    = 0x27
    ONE     = 0x1E
    TWO     = 0x1F
    THREE   = 0x20
    FOUR    = 0x21
    FIVE    = 0x22
    SIX     = 0x23
    SEVEN   = 0x24
    EIGHT   = 0x25
    NINE    = 0x26
    ESC     = 0x29
    SPACE   = 0x2C
    TAB     = 0x2B
    # fmt: on


class MouseButtonMetaEnum(EnumMeta):
    keycode_value_cache: Set[int] = None

    # Override 'in' keyword to check whether the specified integer exists in 'MouseButton'.
    def __contains__(cls, value) -> bool:
        if MouseButtonMetaEnum.keycode_value_cache == None:
            MouseButtonMetaEnum.keycode_value_cache = set(MouseButton)
        return value in MouseButtonMetaEnum.keycode_value_cache


class MouseButton(IntEnum, metaclass=MouseButtonMetaEnum):
    """
    Mouse buttons available to control habitat-hitl.
    """

    # fmt: off
    LEFT   = 0
    RIGHT  = 1
    MIDDLE = 2
    # fmt: on


# On headless systems, we may be unable to import magnum.platform.glfw.Application.
try:
    from magnum.platform.glfw import Application

    magnum_enabled = True
except ImportError:
    print(
        "GuiInput warning: Failed to magnum.platform.glfw. Falling back to agnostic implementation for use with headless server. Local keyboard/mouse input won't work."
    )
    magnum_enabled = False
if magnum_enabled:
    magnum_keymap: Dict[Application.Key, KeyCode] = {
        # fmt: off
        Application.Key.A       : KeyCode.A     ,
        Application.Key.B       : KeyCode.B     ,
        Application.Key.C       : KeyCode.C     ,
        Application.Key.D       : KeyCode.D     ,
        Application.Key.E       : KeyCode.E     ,
        Application.Key.F       : KeyCode.F     ,
        Application.Key.G       : KeyCode.G     ,
        Application.Key.H       : KeyCode.H     ,
        Application.Key.I       : KeyCode.I     ,
        Application.Key.J       : KeyCode.J     ,
        Application.Key.K       : KeyCode.K     ,
        Application.Key.L       : KeyCode.L     ,
        Application.Key.M       : KeyCode.M     ,
        Application.Key.N       : KeyCode.N     ,
        Application.Key.O       : KeyCode.O     ,
        Application.Key.P       : KeyCode.P     ,
        Application.Key.Q       : KeyCode.Q     ,
        Application.Key.R       : KeyCode.R     ,
        Application.Key.S       : KeyCode.S     ,
        Application.Key.T       : KeyCode.T     ,
        Application.Key.U       : KeyCode.U     ,
        Application.Key.V       : KeyCode.V     ,
        Application.Key.W       : KeyCode.W     ,
        Application.Key.X       : KeyCode.X     ,
        Application.Key.Y       : KeyCode.Y     ,
        Application.Key.Z       : KeyCode.Z     ,
        Application.Key.ZERO    : KeyCode.ZERO  ,
        Application.Key.ONE     : KeyCode.ONE   ,
        Application.Key.TWO     : KeyCode.TWO   ,
        Application.Key.THREE   : KeyCode.THREE ,
        Application.Key.FOUR    : KeyCode.FOUR  ,
        Application.Key.FIVE    : KeyCode.FIVE  ,
        Application.Key.SIX     : KeyCode.SIX   ,
        Application.Key.SEVEN   : KeyCode.SEVEN ,
        Application.Key.EIGHT   : KeyCode.EIGHT ,
        Application.Key.NINE    : KeyCode.NINE  ,
        Application.Key.ESC     : KeyCode.ESC   ,
        Application.Key.SPACE   : KeyCode.SPACE ,
        Application.Key.TAB     : KeyCode.TAB   ,
        # fmt: on
    }

    magnum_mouse_keymap: Dict[Application.Pointer, MouseButton] = {
        # fmt: off
        Application.Pointer.MOUSE_LEFT   : MouseButton.LEFT  ,
        Application.Pointer.MOUSE_RIGHT  : MouseButton.RIGHT ,
        Application.Pointer.MOUSE_MIDDLE : MouseButton.MIDDLE,
        # fmt: on
    }


class MagnumKeyConverter:
    def convert_key(key: Any) -> Optional[KeyCode]:
        if magnum_enabled and key in magnum_keymap:
            return magnum_keymap[key]
        return None

    def convert_mouse_button(button: Any) -> Optional[MouseButton]:
        if magnum_enabled and button in magnum_mouse_keymap:
            return magnum_mouse_keymap[button]
        return None
