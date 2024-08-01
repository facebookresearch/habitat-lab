from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Dict, Final, List, Optional

from habitat_hitl.core.user_mask import Mask, Users

if TYPE_CHECKING:
    from habitat_hitl.core.client_message_manager import ClientMessageManager
    from habitat_hitl.core.remote_client_state import RemoteClientState

Color = Optional[List[float]]

AUTO: Final[str] = ""


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
class UISeparator(UIElement):
    """
    Horizontal or vertical line that helps separate content sections.
    """


@dataclass
class UISpacer(UIElement):
    """
    Horizontal or vertical empty space.
    """

    size: float


class HorizontalAlignment(IntEnum):
    LEFT = 0
    CENTER = 1
    RIGHT = 2


class VerticalAlignment(IntEnum):
    TOP = 0
    CENTER = 1
    BOTTOM = 2


@dataclass
class UICanvasUpdate:
    """
    Set of UI updates for a specific canvas.
    """

    clear: bool
    elements: Optional[List[UIElementUpdate]]


@dataclass
class UIElementUpdate:
    """
    UI element to be updated or created.
    """

    canvasProperties: Optional[UICanvas]
    label: Optional[UILabel]
    toggle: Optional[UIToggle]
    button: Optional[UIButton]
    listItem: Optional[UIListItem]
    separator: Optional[UISeparator]
    spacer: Optional[UISpacer]

def _create_default_canvases() -> Dict[str, List[UIElement]]:
    return {
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

def _create_default_user_canvases(user_count: int) -> List[Dict[str, List[UIElement]]]:
    user_canvases: List[Dict[str, List[UIElement]]] = []
    for _ in range(user_count):
        user_canvases.append(_create_default_canvases())
    return user_canvases

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
        self._user_canvases = _create_default_user_canvases(users.max_user_count)

    def update_canvas(
        self, canvas_uid: str, destination_mask: Mask
    ) -> UIContext:
        return UIContext(
            canvas_uid=canvas_uid,
            destination_mask=destination_mask,
            manager=self,
        )

    def commit_canvas_content(
        self,
        canvas_uid: str,
        ui_elements: Dict[str, UIElement],
        destination_mask: Mask,
    ):
        for user_index in self._users.indices(destination_mask):
            cached_elements = self._user_canvases[user_index].get(
                canvas_uid, {}
            )
            self._user_canvases[user_index][canvas_uid] = ui_elements

            def is_canvas_dirty() -> bool:
                if len(cached_elements) != len(ui_elements):
                    return True
                else:
                    return any(
                        uid not in cached_elements
                        or element != cached_elements[uid]
                        for uid, element in ui_elements.items()
                    )

            if is_canvas_dirty():
                dirty_elements: List[UIElementUpdate] = []

                for element in ui_elements.values():
                    dirty_elements.append(
                        UIElementUpdate(
                            canvasProperties=element
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
                            separator=element
                            if isinstance(element, UISeparator)
                            else None,
                            spacer=element
                            if isinstance(element, UISpacer)
                            else None,
                        )
                    )

                # Submit the canvas update.
                canvas_update = UICanvasUpdate(
                    clear=True,
                    elements=dirty_elements,
                )
                self._client_message_manager.update_ui_canvas(
                    canvas_uid=canvas_uid,
                    canvas_update=canvas_update,
                    destination_mask=Mask.from_index(user_index),
                )

    def is_button_pressed(self, uid: str, user_index: int) -> bool:
        return self._client_state.ui_button_pressed(user_index, uid)

    def clear_canvas(self, canvas_uid: str, destination_mask: Mask):
        self._client_message_manager.update_ui_canvas(
            canvas_uid=canvas_uid,
            canvas_update=UICanvasUpdate(
                clear=True,
                elements=None,
            ),
            destination_mask=destination_mask,
        )
        for user_index in self._users.indices(destination_mask):
            self._user_canvases[user_index][canvas_uid].clear()

    def clear_all_canvases(self, destination_mask: Mask):
        for user_index in self._users.indices(destination_mask):
            for canvas_uid in self._user_canvases[user_index].keys():
                self.clear_canvas(canvas_uid, Mask.from_index(user_index))

    def reset(self):
        # If users are connected, clear their UI.
        self.clear_all_canvases(Mask.ALL)

        # Reset internal state.
        self._user_canvases = _create_default_user_canvases(self._users.max_user_count)


class UIContext:
    def __init__(
        self, canvas_uid: str, destination_mask: Mask, manager: UIManager
    ):
        self._canvas_uid = canvas_uid
        self._manager = manager
        self._destination_mask = destination_mask
        self._ui_elements: Dict[str, UIElement] = {}

    def update_element(self, element: UIElement):
        if element.uid == AUTO:
            element.uid = self._generate_uid()
        self._ui_elements[element.uid] = element

    def canvas_properties(
        self,
        padding: int = 0,
        background_color: Optional[List[float]] = None,
    ) -> None:
        self.update_element(
            UICanvas(
                uid=self._canvas_uid,
                padding=padding,
                backgroundColor=background_color,
            )
        )

    def label(
        self,
        uid: str = AUTO,
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
        uid: str = AUTO,
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
        uid: str = AUTO,
        toggled: bool = False,
        text_false: str = "",
        text_true: str = "",
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
        uid: str = AUTO,
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

    def separator(self, uid: str = AUTO):
        self.update_element(UISeparator(uid=uid))

    def spacer(
        self,
        uid: str = AUTO,
        size: float = 24,
    ):
        self.update_element(UISpacer(uid=uid, size=size))

    def _generate_uid(self) -> str:
        return f"{self._canvas_uid}_{len(self._ui_elements)}"

    def __enter__(self) -> UIContext:
        return self

    def __exit__(self, exception_type, _exception_val, _trace):
        self._manager.commit_canvas_content(
            self._canvas_uid, self._ui_elements, self._destination_mask
        )
        return exception_type == None
