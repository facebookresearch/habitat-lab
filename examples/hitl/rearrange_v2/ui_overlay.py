#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, cast

from habitat.sims.habitat_simulator.object_state_machine import (
    BooleanObjectState,
    ObjectStateSpec,
)
from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.core.ui_elements import HorizontalAlignment
from habitat_hitl.core.user_mask import Mask

FONT_SIZE_LARGE: Final[int] = 32
FONT_SIZE_SMALL: Final[int] = 24
FONT_COLOR_WARNING: Final[List[float]] = [1.0, 0.75, 0.5, 1.0]
PANEL_BACKGROUND_COLOR: Final[List[float]] = [0.7, 0.7, 0.7, 0.3]
TOGGLE_COLOR_AVAILABLE: Final[List[float]] = [0.1, 0.8, 0.8, 1.0]
TOGGLE_COLOR_RECENTLY_CHANGED: Final[List[float]] = [0.1, 0.8, 0.1, 1.0]
SPACE_SIZE = 6


@dataclass
class ObjectStateControl:
    """
    Collection of information that allows for displaying and manipulating object states.
    """

    spec: ObjectStateSpec
    value: bool
    enabled: bool
    available: bool
    callback: Optional[
        Callable[
            [str, str, Any], None  # object_handle  # state_name  # state_value
        ]
    ]
    tooltip: Optional[str]
    recently_changed: bool


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

    def update_hovered_object_info_panel(
        self,
        object_category_name: Optional[str],
        object_states: List[Tuple[str, str]],
        primary_region_name: Optional[str],
    ):
        """
        Update the panel that shows information about the hovered object.
        """
        manager = self._ui_manager
        with manager.update_canvas("bottom", self._dest_mask) as ctx:
            if object_category_name is None:
                return

            ctx.canvas_properties(
                padding=12, background_color=PANEL_BACKGROUND_COLOR
            )

            title = self._title_str(object_category_name)

            ctx.label(
                text=title,
                font_size=FONT_SIZE_LARGE,
                horizontal_alignment=HorizontalAlignment.CENTER,
            )

            ctx.spacer(size=SPACE_SIZE)

            region_name = (
                self._title_str(primary_region_name)
                if primary_region_name is not None
                else "None"
            )
            ctx.list_item(
                text_left="Room",
                text_right=region_name,
                font_size=FONT_SIZE_SMALL,
            )

            def create_list_item(left: str, right: str):
                ctx.list_item(
                    text_left=left,
                    text_right=right,
                    font_size=FONT_SIZE_SMALL,
                )

            for state in object_states:
                create_list_item(state[0], state[1])

    def update_selected_object_panel(
        self,
        object_category_name: Optional[str],
        object_state_controls: List[ObjectStateControl],
        primary_region_name: Optional[str],
        contextual_info: List[Tuple[str, Optional[List[float]]]],
    ):
        """
        Draw a panel that shows information about the selected object.
        Allow for editing object states.
        """
        manager = self._ui_manager
        with manager.update_canvas("bottom_left", self._dest_mask) as ctx:
            if object_category_name is None:
                return

            ctx.canvas_properties(
                padding=12, background_color=PANEL_BACKGROUND_COLOR
            )

            title = self._title_str(object_category_name)

            ctx.label(
                text=title,
                font_size=FONT_SIZE_LARGE,
                horizontal_alignment=HorizontalAlignment.CENTER,
            )

            for info_label in contextual_info:
                ctx.label(
                    text=info_label[0],
                    font_size=FONT_SIZE_SMALL,
                    horizontal_alignment=HorizontalAlignment.CENTER,
                    color=info_label[1],
                )

            ctx.spacer(size=SPACE_SIZE)

            region_name = (
                self._title_str(primary_region_name)
                if primary_region_name is not None
                else "None"
            )
            ctx.list_item(
                text_left="Room",
                text_right=region_name,
                font_size=FONT_SIZE_SMALL,
            )

            def create_toggle(osc: ObjectStateControl) -> str:
                spec = cast(BooleanObjectState, osc.spec)
                item_key = f"select_{spec.name}"
                color = None
                if osc.recently_changed:
                    color = TOGGLE_COLOR_RECENTLY_CHANGED
                elif osc.available:
                    color = TOGGLE_COLOR_AVAILABLE
                ctx.toggle(
                    uid=item_key,
                    text_false=spec.display_name_false,
                    text_true=spec.display_name_true,
                    toggled=osc.value,
                    enabled=osc.enabled and osc.available,
                    tooltip=osc.tooltip,
                    color=color,
                )
                return item_key

            for osc in object_state_controls:
                button_key = create_toggle(osc)
                if osc.callback is not None:
                    self._buttons[button_key] = osc.callback

    @staticmethod
    def _title_str(string: str):
        """Convert 'snake_case' to 'Title Case'."""
        return (
            string.replace("_", " ")
            .replace("-", " ")
            .replace(".", " ")
            .title()
        )
