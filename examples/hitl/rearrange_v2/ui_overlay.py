#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import textwrap
from typing import Callable, Dict, Final, List, Optional, Tuple

from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.core.ui_elements import HorizontalAlignment
from habitat_hitl.core.user_mask import Mask

FONT_SIZE_LARGE: Final[int] = 32
FONT_SIZE_SMALL: Final[int] = 24
FONT_COLOR_WARNING: Final[List[float]] = [1.0, 0.75, 0.5, 1.0]
PANEL_BACKGROUND_COLOR: Final[List[float]] = [0.7, 0.7, 0.7, 0.3]
SPACE_SIZE = 6


class UIOverlay:
    """
    Overlay GUI for a specific user.
    """

    def __init__(
        self,
        app_service: AppService,
        user_index: int,
    ):
        self._ui_manager = app_service.ui_manager
        self._buttons: Dict[str, Callable] = {}
        self._user_index = user_index
        self._dest_mask: Mask = Mask.from_index(user_index)

    def update(self):
        manager = self._ui_manager
        for uid, callback in self._buttons.items():
            if manager.is_button_pressed(uid, self._user_index):
                callback()

    def reset(self):
        self._ui_manager.clear_all_canvases(self._dest_mask)
        self._buttons.clear()

    def update_instructions_panel(
        self,
        instructions: Optional[str],
        warning_text: Optional[str],
        is_help_shown: bool,
    ):
        """
        Update the HITL task instruction UI panel.
        """
        manager = self._ui_manager
        has_instructions = instructions is not None and len(instructions) > 0
        has_warning_text = warning_text is not None and len(warning_text) > 0
        with manager.update_canvas("top_left", self._dest_mask) as ctx:
            if has_instructions:
                ctx.canvas_properties(
                    padding=12, background_color=PANEL_BACKGROUND_COLOR
                )

                if is_help_shown:
                    ctx.label(
                        text="Instructions",
                        font_size=FONT_SIZE_LARGE,
                        horizontal_alignment=HorizontalAlignment.LEFT,
                    )

                    ctx.separator()

                multiline_instructions = textwrap.fill(
                    instructions,
                    width=70,
                    break_long_words=False,
                    break_on_hyphens=True,
                )

                ctx.label(
                    text=multiline_instructions,
                    font_size=FONT_SIZE_SMALL,
                    horizontal_alignment=HorizontalAlignment.LEFT,
                )

            if has_warning_text:
                if has_instructions:
                    ctx.spacer(size=SPACE_SIZE)

                multiline_warning = textwrap.fill(
                    warning_text,
                    width=70,
                    break_long_words=False,
                    break_on_hyphens=True,
                )

                ctx.label(
                    text=multiline_warning,
                    font_size=FONT_SIZE_SMALL,
                    bold=True,
                    horizontal_alignment=HorizontalAlignment.LEFT,
                    color=FONT_COLOR_WARNING,
                )

    def update_controls_panel(
        self,
        controls: Optional[List[Tuple[str, str]]],
    ):
        """
        Update the HITL controls UI panel.
        """
        manager = self._ui_manager
        with manager.update_canvas("top_right", self._dest_mask) as ctx:
            if controls is None:
                ctx.canvas_properties(
                    padding=12, background_color=PANEL_BACKGROUND_COLOR
                )

                ctx.list_item(
                    text_left="H",
                    text_right="Show Help",
                    font_size=FONT_SIZE_SMALL,
                )

            else:
                ctx.canvas_properties(
                    padding=12, background_color=PANEL_BACKGROUND_COLOR
                )

                ctx.label(
                    text="Controls",
                    font_size=FONT_SIZE_LARGE,
                    horizontal_alignment=HorizontalAlignment.RIGHT,
                )

                for control in controls:
                    ctx.list_item(
                        text_left=control[0],
                        text_right=control[1],
                        font_size=FONT_SIZE_SMALL,
                    )
