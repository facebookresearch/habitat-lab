#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import textwrap
from typing import TYPE_CHECKING, Any, Callable, Dict, Final, List, Optional, Tuple

from habitat.sims.habitat_simulator.object_state_machine import ObjectStateSpec
from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.core.ui_elements import HorizontalAlignment, UIManager
import magnum as mn
from world import World

FONT_SIZE_LARGE: Final[int] = 32
FONT_SIZE_SMALL: Final[int] = 24

@dataclass
class ObjectStateControl:
    spec: ObjectStateSpec
    value: bool
    enabled: bool
    available: bool
    callback: Callable [[
        str, # object_handle
        str, # state_name
        Any # state_value
        ], None]

class UIOverlay:
    """
    User interface overlay.
    """

    def __init__(
        self,
        app_service: AppService,
        user_index: int,
    ):
        self._ui_manager = UIManager(app_service, user_index)
        self._buttons: Dict[str, Callable] = {}

    def update(self):
        manager = self._ui_manager
        for uid, callback in self._buttons.items():
            if manager.is_button_pressed(uid):
                callback()

    def reset(self):
        self._ui_manager.clear_all_canvases()
        self._buttons.clear()

    def update_instructions_panel(
        self,
        instructions: Optional[str],
        status_text: Optional[str],
    ):
        manager = self._ui_manager
        has_instructions = instructions is not None and len(instructions) > 0
        has_status_text = status_text is not None and len(status_text) > 0
        with manager.update_canvas("top_left") as ctx:
            if has_instructions:            
                ctx.label(
                    uid="instr_title",
                    text="Instructions",
                    font_size=FONT_SIZE_LARGE,
                    horizontal_alignment=HorizontalAlignment.LEFT
                )

                multiline_instructions = textwrap.fill(
                    instructions,
                    width=70,
                    break_long_words=False,
                    break_on_hyphens=True
                )

                ctx.label(
                    uid="instr_content",
                    text=multiline_instructions,
                    font_size=FONT_SIZE_SMALL,
                    horizontal_alignment=HorizontalAlignment.LEFT
                )

            if has_status_text:
                multiline_status = textwrap.fill(
                    status_text,
                    width=70,
                    break_long_words=False,
                    break_on_hyphens=True
                )

                ctx.label(
                    uid="instr_status",
                    text=multiline_status,
                    font_size=FONT_SIZE_SMALL,
                    bold=True,
                    horizontal_alignment=HorizontalAlignment.LEFT,
                    color=[1.0, 0.75, 0.5, 1.0]
                )

    def update_controls_panel(
        self,
        controls: Optional[List[Tuple[str, str]]],
    ):
        manager = self._ui_manager
        with manager.update_canvas("top_right") as ctx:
            if controls is None:
                return

            ctx.label(
                "ctrl_title",
                text="Controls",
                font_size=FONT_SIZE_LARGE,
                horizontal_alignment=HorizontalAlignment.RIGHT
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

            for tuple in controls:
                create_list_item(tuple[0], tuple[1])

    def update_hovered_object_info_panel(
        self,
        object_category_name: Optional[str],
        object_states: List[Tuple[str, str]],
    ):
        manager = self._ui_manager
        with manager.update_canvas("bottom_left") as ctx:
            if object_category_name is None:
                return

            title = _display_str(object_category_name)

            ctx.label(
                "hover_title",
                text=title,
                font_size=FONT_SIZE_LARGE,
                horizontal_alignment=HorizontalAlignment.LEFT
            )

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

            for tuple in object_states:
                create_list_item(tuple[0], tuple[1])

    def update_selected_object_panel(
        self,
        object_category_name: Optional[str],
        toggles: List[ObjectStateControl],
        canvas_position: Optional[mn.Vector3],
    ):
        manager = self._ui_manager
        with manager.update_canvas("floating") as ctx:
            if object_category_name is None:
                return
            
            title = _display_str(object_category_name)

            ctx.label(
                "select_title",
                text=title,
                font_size=FONT_SIZE_LARGE,
                horizontal_alignment=HorizontalAlignment.CENTER
            )

            def create_toggle(toggle: ObjectStateControl) -> str:
                spec = toggle.spec
                item_key = f"select_{spec.name}"
                ctx.toggle(
                    item_key,
                    text_false=spec.display_name_false,
                    text_true=spec.display_name_true,
                    toggled=toggle.value,
                    enabled=toggle.enabled and toggle.available,
                )
                return item_key

            for toggle in toggles:
                button_key = create_toggle(toggle)
                self._buttons[button_key] = toggle.callback

        if canvas_position is not None:
            manager.move_canvas("floating", canvas_position)
    
def _display_str(string: str):
    """Convert 'internal_case' to 'Title Case'."""
    return string.replace("_", " ").replace("-", " ").replace(".", " ").title()