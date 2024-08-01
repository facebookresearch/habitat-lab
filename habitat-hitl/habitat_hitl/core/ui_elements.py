from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Dict, List, Optional

import magnum as mn

from habitat_hitl.core.user_mask import Mask, Users

if TYPE_CHECKING:
    from habitat_hitl.core.client_message_manager import ClientMessageManager
    from habitat_hitl.core.remote_client_state import RemoteClientState

Color = Optional[List[float]]


@dataclass
class UIElement:
    """
    Base class for all networked UI elements.
    """

    uid: str


@dataclass
class UICanvas(UIElement):
    """
    Canvas properties
    """

    padding: int
    backgroundColor: Color


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
    tooltip: str


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

    canvasUid: str
    canvas: Optional[UICanvas]
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

    def __init__(
        self,
        users: Users,
        client_state: "RemoteClientState",
        client_message_manager: "ClientMessageManager",
    ):
        self._client_state = client_state
        self._client_message_manager = client_message_manager
        self._users = users

        self._user_canvases: List[Dict[str, Dict[str, UIElement]]] = []
        for _ in range(self._users.max_user_count):
            # TODO: Canvases are currently predefined.
            self._user_canvases.append(
                {
                    "top_left": {},
                    "top": {},
                    "top_right": {},
                    "left": {},
                    "center": {},
                    "right": {},
                    "bottom_left": {},
                    "bottom": {},
                    "bottom_right": {},
                    "tooltip": {},
                }
            )

    def update_canvas(
        self, canvas_name: str, destination_mask: Mask
    ) -> UIContext:
        return UIContext(
            canvas_name=canvas_name,
            destination_mask=destination_mask,
            manager=self,
        )

    def commit_canvas_content(
        self,
        canvas_name: str,
        ui_elements: Dict[str, UIElement],
        destination_mask: Mask,
    ):
        for user_index in self._users.indices(destination_mask):
            canvas_elements = self._user_canvases[user_index].get(
                canvas_name, {}
            )
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
                            canvasUid=canvas_name,
                            canvas=element
                            if isinstance(element, UICanvas)
                            else None,
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
                    destination_mask=Mask.from_index(user_index),
                )

            # If the canvas is dirty, clear it.
            if canvas_dirty:
                self.clear_canvas(canvas_name, Mask.from_index(user_index))

            # Register UI elements.
            for uid, element in ui_elements.items():
                self._user_canvases[user_index][canvas_name][uid] = element

    def is_button_pressed(self, uid: str, user_index: int) -> bool:
        return self._client_state.ui_button_pressed(user_index, uid)

    def clear_canvas(self, canvas_name: str, destination_mask: Mask):
        self._client_message_manager.clear_canvas(
            canvas_name, destination_mask=destination_mask
        )
        for user_index in self._users.indices(destination_mask):
            self._user_canvases[user_index][canvas_name].clear()

    def clear_all_canvases(self, destination_mask: Mask):
        for user_index in self._users.indices(destination_mask):
            for canvas_name in self._user_canvases[user_index].keys():
                self.clear_canvas(canvas_name, Mask.from_index(user_index))

    def move_canvas(
        self, canvas: str, world_position: mn.Vector3, destination_mask: Mask
    ):
        world_pos: List[float] = [
            world_position.x,
            world_position.y,
            world_position.z,
        ]
        self._client_message_manager.move_canvas(
            canvas,
            world_pos,
            destination_mask=Mask.from_index(destination_mask),
        )


class UIContext:
    def __init__(
        self, canvas_name: str, destination_mask: Mask, manager: UIManager
    ):
        self._canvas_name = canvas_name
        self._manager = manager
        self._destination_mask = destination_mask
        self._ui_updates: List[UIUpdate] = []
        self._ui_elements: Dict[str, UIElement] = {}

    def update_element(self, element: UIElement):
        self._ui_elements[element.uid] = element

    def canvas(
        self,
        padding: int = 0,
        background_color: Optional[List[float]] = None,
    ) -> None:
        self.update_element(
            UICanvas(
                uid=self._canvas_name,
                padding=padding,
                backgroundColor=background_color,
            )
        )

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
        tooltip: Optional[str] = None,
    ) -> None:
        self.update_element(
            UIToggle(
                uid=uid,
                enabled=enabled,
                color=color,
                toggled=toggled,
                textFalse=text_false,
                textTrue=text_true,
                tooltip=tooltip,
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
            self._canvas_name, self._ui_elements, self._destination_mask
        )
        return exception_type == None
