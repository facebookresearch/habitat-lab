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
    magnum_keymap: Dict[Application.KeyEvent.Key, KeyCode] = {
        # fmt: off
        Application.KeyEvent.Key.A       : KeyCode.A     ,
        Application.KeyEvent.Key.B       : KeyCode.B     ,
        Application.KeyEvent.Key.C       : KeyCode.C     ,
        Application.KeyEvent.Key.D       : KeyCode.D     ,
        Application.KeyEvent.Key.E       : KeyCode.E     ,
        Application.KeyEvent.Key.F       : KeyCode.F     ,
        Application.KeyEvent.Key.G       : KeyCode.G     ,
        Application.KeyEvent.Key.H       : KeyCode.H     ,
        Application.KeyEvent.Key.I       : KeyCode.I     ,
        Application.KeyEvent.Key.J       : KeyCode.J     ,
        Application.KeyEvent.Key.K       : KeyCode.K     ,
        Application.KeyEvent.Key.L       : KeyCode.L     ,
        Application.KeyEvent.Key.M       : KeyCode.M     ,
        Application.KeyEvent.Key.N       : KeyCode.N     ,
        Application.KeyEvent.Key.O       : KeyCode.O     ,
        Application.KeyEvent.Key.P       : KeyCode.P     ,
        Application.KeyEvent.Key.Q       : KeyCode.Q     ,
        Application.KeyEvent.Key.R       : KeyCode.R     ,
        Application.KeyEvent.Key.S       : KeyCode.S     ,
        Application.KeyEvent.Key.T       : KeyCode.T     ,
        Application.KeyEvent.Key.U       : KeyCode.U     ,
        Application.KeyEvent.Key.V       : KeyCode.V     ,
        Application.KeyEvent.Key.W       : KeyCode.W     ,
        Application.KeyEvent.Key.X       : KeyCode.X     ,
        Application.KeyEvent.Key.Y       : KeyCode.Y     ,
        Application.KeyEvent.Key.Z       : KeyCode.Z     ,
        Application.KeyEvent.Key.ZERO    : KeyCode.ZERO  ,
        Application.KeyEvent.Key.ONE     : KeyCode.ONE   ,
        Application.KeyEvent.Key.TWO     : KeyCode.TWO   ,
        Application.KeyEvent.Key.THREE   : KeyCode.THREE ,
        Application.KeyEvent.Key.FOUR    : KeyCode.FOUR  ,
        Application.KeyEvent.Key.FIVE    : KeyCode.FIVE  ,
        Application.KeyEvent.Key.SIX     : KeyCode.SIX   ,
        Application.KeyEvent.Key.SEVEN   : KeyCode.SEVEN ,
        Application.KeyEvent.Key.EIGHT   : KeyCode.EIGHT ,
        Application.KeyEvent.Key.NINE    : KeyCode.NINE  ,
        Application.KeyEvent.Key.ESC     : KeyCode.ESC   ,
        Application.KeyEvent.Key.SPACE   : KeyCode.SPACE ,
        Application.KeyEvent.Key.TAB     : KeyCode.TAB   ,
        # fmt: on
    }

    magnum_mouse_keymap: Dict[Application.KeyEvent.Key, MouseButton] = {
        # fmt: off
        Application.MouseEvent.Button.LEFT   : MouseButton.LEFT  ,
        Application.MouseEvent.Button.RIGHT  : MouseButton.RIGHT ,
        Application.MouseEvent.Button.MIDDLE : MouseButton.MIDDLE,
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
