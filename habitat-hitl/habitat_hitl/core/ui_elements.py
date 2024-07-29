from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Dict, List, Optional

import magnum as mn

from habitat_hitl.core.user_mask import Mask

if TYPE_CHECKING:
    from habitat_hitl.app_states.app_service import AppService


Color = Optional[List[float]]


@dataclass
class UIElement:
    """
    Base class for all networked UI elements.
    """

    uid: str


@dataclass
class UILabel(UIElement):
    """
    Text label.
    """

    text: str
    horizontalAlignment: int
    fontSize: int
    bold: bool
    color: Color


@dataclass
class UIToggle(UIElement):
    """
    Toggle button with two labels.
    """

    enabled: bool
    toggled: bool
    textFalse: str
    textTrue: str
    color: Color


@dataclass
class UIListItem(UIElement):
    """
    List item with two labels.
    """

    textLeft: str
    textRight: str
    fontSize: int
    color: Color


@dataclass
class UIButton(UIElement):
    """
    Clickable button with a text label.
    """

    enabled: bool
    text: str
    color: Color


@dataclass
class UIUpdate:
    """
    Set of UI updates for a specific canvas.
    """

    canvas: str
    label: Optional[UILabel]
    toggle: Optional[UIToggle]
    button: Optional[UIButton]
    listItem: Optional[UIListItem]


class HorizontalAlignment(IntEnum):
    LEFT = 0
    CENTER = 1
    RIGHT = 2


class VerticalAlignment(IntEnum):
    TOP = 0
    CENTER = 1
    BOTTOM = 2


class UIManager:
    """
    Helper class for handling UI using UIElements.
    """

    def __init__(self, app_service: AppService, user_index: int):
        self._client_state = app_service.remote_client_state
        self._client_message_manager = app_service.client_message_manager
        self._user_index = user_index
        self._users = app_service.users

        # TODO: Canvases are currently predefined.
        self._canvases: Dict[str, Dict[str, UIElement]] = {
            "top_left": {},
            "top_right": {},
            "bottom_left": {},
            "bottom_right": {},
            "floating": {},
        }

    def update_canvas(self, canvas_name: str) -> UIContext:
        assert canvas_name in self._canvases
        return UIContext(
            canvas_name=canvas_name,
            user_index=self._user_index,
            manager=self,
        )

    def commit_canvas_content(
        self, canvas_name: str, ui_elements: Dict[str, UIElement]
    ):
        assert canvas_name in self._canvases

        canvas_elements = self._canvases[canvas_name]
        canvas_dirty = len(canvas_elements) != len(ui_elements)
        dirty_elements: List[UIUpdate] = []

        for uid, element in ui_elements.items():
            if not canvas_dirty and (
                uid not in canvas_elements
                or type(element) != type(canvas_elements[uid])
            ):
                canvas_dirty = True

            # If the element has changed or the canvas is dirty, update the element.
            if canvas_dirty or element != canvas_elements[uid]:
                dirty_elements.append(
                    UIUpdate(
                        canvas=canvas_name,
                        label=element
                        if isinstance(element, UILabel)
                        else None,
                        toggle=element
                        if isinstance(element, UIToggle)
                        else None,
                        button=element
                        if isinstance(element, UIButton)
                        else None,
                        listItem=element
                        if isinstance(element, UIListItem)
                        else None,
                    )
                )

        # Submit the UI updates.
        for dirty_element in dirty_elements:
            self._client_message_manager.update_ui(
                ui_update=dirty_element,
                destination_mask=Mask.from_index(self._user_index),
            )

        # If the canvas is dirty, clear it.
        if canvas_dirty:
            self.clear_canvas(canvas_name)

        # Register UI elements.
        for uid, element in ui_elements.items():
            self._canvases[canvas_name][uid] = element

    def is_button_pressed(self, uid: str) -> bool:
        return self._client_state.ui_button_pressed(self._user_index, uid)

    def clear_canvas(self, canvas_name: str):
        assert canvas_name in self._canvases
        self._client_message_manager.clear_canvas(
            canvas_name, destination_mask=Mask.from_index(self._user_index)
        )
        self._canvases[canvas_name].clear()

    def clear_all_canvases(self):
        for canvas_name in self._canvases.keys():
            self.clear_canvas(canvas_name)

    def move_canvas(self, canvas: str, world_position: mn.Vector3):
        world_pos: List[float] = [
            world_position.x,
            world_position.y,
            world_position.z,
        ]
        self._client_message_manager.move_canvas(
            canvas,
            world_pos,
            destination_mask=Mask.from_index(self._user_index),
        )


class UIContext:
    def __init__(self, canvas_name: str, user_index: int, manager: UIManager):
        self._canvas_name = canvas_name
        self._manager = manager
        self._user_index = user_index
        self._ui_updates: List[UIUpdate] = []
        self._ui_elements: Dict[str, UIElement] = {}

    def update_element(self, element: UIElement):
        self._ui_elements[element.uid] = element

    def label(
        self,
        uid: str,
        text: str = "",
        font_size: int = 24,
        bold: bool = False,
        horizontal_alignment: HorizontalAlignment = HorizontalAlignment.LEFT,
        color: Optional[List[float]] = None,
    ) -> None:
        self.update_element(
            UILabel(
                uid=uid,
                text=text,
                fontSize=font_size,
                bold=bold,
                horizontalAlignment=horizontal_alignment,
                color=color,
            )
        )

    def list_item(
        self,
        uid: str,
        text_left: str = "",
        text_right: str = "",
        font_size: int = 24,
        color: Optional[List[float]] = None,
    ) -> None:
        self.update_element(
            UIListItem(
                uid=uid,
                textLeft=text_left,
                textRight=text_right,
                fontSize=font_size,
                color=color,
            )
        )

    def toggle(
        self,
        uid: str,
        toggled: bool,
        text_false: str,
        text_true: str,
        enabled: bool = True,
        color: Optional[List[float]] = None,
    ) -> None:
        self.update_element(
            UIToggle(
                uid=uid,
                enabled=enabled,
                color=color,
                toggled=toggled,
                textFalse=text_false,
                textTrue=text_true,
            )
        )

    def button(
        self,
        uid: str,
        text: str = "",
        enabled: bool = True,
        color: Optional[List[float]] = None,
    ) -> None:
        self.update_element(
            UIButton(
                uid=uid,
                text=text,
                enabled=enabled,
                color=color,
            )
        )

    def __enter__(self) -> UIContext:
        return self

    def __exit__(self, exception_type, _exception_val, _trace):
        self._manager.commit_canvas_content(
            self._canvas_name, self._ui_elements
        )
        return exception_type == None
