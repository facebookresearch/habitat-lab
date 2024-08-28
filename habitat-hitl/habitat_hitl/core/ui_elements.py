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

    @staticmethod
    def create(e: UIElement) -> UIElementUpdate:
        return UIElementUpdate(
            canvasProperties=e if isinstance(e, UICanvas) else None,
            label=e if isinstance(e, UILabel) else None,
            toggle=e if isinstance(e, UIToggle) else None,
            button=e if isinstance(e, UIButton) else None,
            listItem=e if isinstance(e, UIListItem) else None,
            separator=e if isinstance(e, UISeparator) else None,
            spacer=e if isinstance(e, UISpacer) else None,
        )


def _create_default_canvases() -> Dict[str, Dict[str, UIElement]]:
    """
    Create a map of available canvases to their content.
    By the default, the canvases are empty.
    """
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


def _create_default_user_canvases(
    user_count: int,
) -> List[Dict[str, Dict[str, UIElement]]]:
    """
    Create a list of canvases per user.
    By the default, the canvases are empty.
    """
    user_canvases: List[Dict[str, Dict[str, UIElement]]] = []
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
        self._user_canvases = _create_default_user_canvases(
            users.max_user_count
        )

    def update_canvas(
        self, canvas_uid: str, destination_mask: Mask
    ) -> UIContext:
        """
        Update a canvas.

        Use this function within a `with` block.
        The canvas update will consolidated and sent to all users in `destination_mask` when the `with` block de-scopes.
        Updating a canvas more than once will overwrite the previous state.

        Example:
        ```
        with ui.update_canvas("center", Mask.ALL) as ctx:
            ctx.label(text="Text")
        ```
        """
        return UIContext(
            canvas_uid=canvas_uid,
            destination_mask=destination_mask,
            manager=self,
        )

    def _commit_canvas_content(
        self,
        canvas_uid: str,
        ui_elements: Dict[str, UIElement],
        destination_mask: Mask,
    ):
        """
        Update a canvas.

        Use `update_canvas()` instead of calling this function directly.
        """
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
                    dirty_elements.append(UIElementUpdate.create(element))

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
        self._user_canvases = _create_default_user_canvases(
            self._users.max_user_count
        )


class UIContext:
    """
    Helper class for writing a UI canvas update.

    Create using `UIManager.update_canvas()`.

    Example:
    ```
    with ui.update_canvas("center", Mask.ALL) as ctx:
        ctx.label(text="Text")
    ```
    """

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
        *,
        padding: int = 0,
        background_color: Optional[List[float]] = None,
    ) -> None:
        """
        Set the properties of the canvas.

        `padding`: Size of the space around the canvas content.
        `background_color`: RGBA color of the canvas background. Transparent by default.
        """
        self.update_element(
            UICanvas(
                uid=self._canvas_uid,
                padding=padding,
                backgroundColor=background_color,
            )
        )

    def label(
        self,
        *,
        uid: str = AUTO,
        text: str = "",
        font_size: int = 24,
        bold: bool = False,
        horizontal_alignment: HorizontalAlignment = HorizontalAlignment.LEFT,
        color: Optional[List[float]] = None,
    ) -> None:
        """
        Create a text label.

        `uid`: Unique identifier for the element. Autogenerated by default.
        `text`: Text content of the label.
        `font_size`: Size of the font.
        `bold`: Whether the text is bold.
        `horizontal_alignment`: Horizontal alignment of the text.
        `color`: RGBA color of the text. White by default.
        """
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
        *,
        uid: str = AUTO,
        text_left: str = "",
        text_right: str = "",
        font_size: int = 24,
        color: Optional[List[float]] = None,
    ) -> None:
        """
        Create a list item with two labels.

        `uid`: Unique identifier for the element. Autogenerated by default.
        `text_left`: Text content of the left label.
        `text_right`: Text content of the right label.
        `font_size`: Size of the text.
        `color`: RGBA color of the text. White by default.
        """
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
        *,
        uid: str = AUTO,
        toggled: bool = False,
        text_false: str = "",
        text_true: str = "",
        enabled: bool = True,
        color: Optional[List[float]] = None,
        tooltip: Optional[str] = None,
    ) -> None:
        """
        Create a toggle with "true" and "false" labels on each side.
        Use 'UIManager.is_button_pressed()' to check whether the toggle was pressed during the frame.

        `uid`: Unique identifier for the element. Autogenerated by default.
        `toggled`: Whether the toggle is activated.
        `text_false`: Text label for the deactivated state.
        `text_true`: Text label for the activated state.
        `enabled`: Whether the toggle state can be changed.
        `color`: RGBA color of the toggle.
        `tooltip`: Tooltip to show when hovering the toggle.
        """
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
        *,
        uid: str = AUTO,
        text: str = "",
        enabled: bool = True,
        color: Optional[List[float]] = None,
    ) -> None:
        """
        Create a button with text content.
        Use 'UIManager.is_button_pressed()' to check whether the button was pressed during the frame.

        `uid`: Unique identifier for the element. Autogenerated by default.
        `text`: Text to display on the button.
        `enabled`: Whether the toggle state can be changed.
        `color`: RGBA color of the button.
        """
        self.update_element(
            UIButton(
                uid=uid,
                text=text,
                enabled=enabled,
                color=color,
            )
        )

    def separator(self, *, uid: str = AUTO):
        """
        Create a horizontal line separator.
        """
        self.update_element(UISeparator(uid=uid))

    def spacer(
        self,
        *,
        uid: str = AUTO,
        size: float = 24,
    ):
        """
        Add an empty space.
        """
        self.update_element(UISpacer(uid=uid, size=size))

    def _generate_uid(self) -> str:
        return f"{self._canvas_uid}_{len(self._ui_elements)}"

    def __enter__(self) -> UIContext:
        return self

    def __exit__(self, exception_type, _exception_val, _trace):
        self._manager._commit_canvas_content(
            self._canvas_uid, self._ui_elements, self._destination_mask
        )
        return exception_type == None
