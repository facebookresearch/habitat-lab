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


@dataclass
class ObjectStateControl:
    spec: ObjectStateSpec
    value: bool
    enabled: bool
    available: bool
    callback: Callable[
        [str, str, Any], None  # object_handle  # state_name  # state_value
    ]
    tooltip: Optional[str]


class UIOverlay:
    """
    User interface overlay.
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
        status_text: Optional[str],
    ):
        manager = self._ui_manager
        has_instructions = instructions is not None and len(instructions) > 0
        has_status_text = status_text is not None and len(status_text) > 0
        with manager.update_canvas("top_left", self._dest_mask) as ctx:
            if has_instructions:
                ctx.canvas_properties(
                    padding=12, background_color=[0.7, 0.7, 0.7, 0.3]
                )

                ctx.label(
                    text="Instructions",
                    font_size=FONT_SIZE_LARGE,
                    horizontal_alignment=HorizontalAlignment.LEFT,
                )

                # TODO: Separator element.
                ctx.list_item()

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

            if has_status_text:
                multiline_status = textwrap.fill(
                    status_text,
                    width=70,
                    break_long_words=False,
                    break_on_hyphens=True,
                )

                ctx.label(
                    text=multiline_status,
                    font_size=FONT_SIZE_SMALL,
                    bold=True,
                    horizontal_alignment=HorizontalAlignment.LEFT,
                    color=[1.0, 0.75, 0.5, 1.0],
                )

    def update_controls_panel(
        self,
        controls: Optional[List[Tuple[str, str]]],
    ):
        manager = self._ui_manager
        with manager.update_canvas("top_right", self._dest_mask) as ctx:
            if controls is None:
                return

            ctx.canvas_properties(
                padding=12, background_color=[0.7, 0.7, 0.7, 0.3]
            )

            ctx.label(
                text="Controls",
                font_size=FONT_SIZE_LARGE,
                horizontal_alignment=HorizontalAlignment.RIGHT,
            )

            current_item_id = 0

            def create_list_item(left: str, right: str):
                nonlocal current_item_id
                item_key = f"ctrl_{current_item_id}"
                ctx.list_item(
                    item_key,
                    text_left=left,
                    text_right=right,
                    font_size=FONT_SIZE_SMALL,
                )
                current_item_id += 1

            for control in controls:
                create_list_item(control[0], control[1])

    def update_hovered_object_info_panel(
        self,
        object_category_name: Optional[str],
        object_states: List[Tuple[str, str]],
    ):
        manager = self._ui_manager
        with manager.update_canvas("bottom", self._dest_mask) as ctx:
            if object_category_name is None:
                return

            ctx.canvas_properties(
                padding=12, background_color=[0.7, 0.7, 0.7, 0.3]
            )

            title = _display_str(object_category_name)

            ctx.label(
                text=title,
                font_size=FONT_SIZE_LARGE,
                horizontal_alignment=HorizontalAlignment.CENTER,
            )

            # TODO: Separator element.
            ctx.list_item()

            current_item_id = 0

            def create_list_item(left: str, right: str):
                nonlocal current_item_id
                item_key = f"hover_{current_item_id}"
                ctx.list_item(
                    item_key,
                    text_left=left,
                    text_right=right,
                    font_size=FONT_SIZE_SMALL,
                )
                current_item_id += 1

            for state in object_states:
                create_list_item(state[0], state[1])

    def update_selected_object_panel(
        self,
        object_category_name: Optional[str],
        toggles: List[ObjectStateControl],
    ):
        manager = self._ui_manager
        with manager.update_canvas("bottom_left", self._dest_mask) as ctx:
            if object_category_name is None:
                return

            color_available = [0.1, 0.8, 0.8, 1.0]

            ctx.canvas_properties(
                padding=12, background_color=[0.3, 0.3, 0.3, 0.7]
            )

            title = _display_str(object_category_name)

            ctx.label(
                text=title,
                font_size=FONT_SIZE_LARGE,
                horizontal_alignment=HorizontalAlignment.CENTER,
            )

            # TODO: Separator element.
            ctx.list_item()

            def create_toggle(toggle: ObjectStateControl) -> str:
                spec = cast(BooleanObjectState, toggle.spec)
                item_key = f"select_{spec.name}"
                ctx.toggle(
                    item_key,
                    text_false=spec.display_name_false,
                    text_true=spec.display_name_true,
                    toggled=toggle.value,
                    enabled=toggle.enabled and toggle.available,
                    tooltip=toggle.tooltip
                    if toggle.available
                    else "Action unavailable.",
                    color=color_available if toggle.available else None,
                )
                return item_key

            for toggle in toggles:
                button_key = create_toggle(toggle)
                self._buttons[button_key] = toggle.callback


def _display_str(string: str):
    """Convert 'internal_case' to 'Title Case'."""
    return string.replace("_", " ").replace("-", " ").replace(".", " ").title()
