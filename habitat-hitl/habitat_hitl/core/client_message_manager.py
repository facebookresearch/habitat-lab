#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import asdict, dataclass
from typing import Any, Dict, Final, List, Optional, Union

import magnum as mn

from habitat_hitl.core.types import Message
from habitat_hitl.core.user_mask import Mask, Users

DEFAULT_NORMAL: Final[List[float]] = [0.0, 1.0, 0.0]
DEFAULT_VIEWPORT_SIZE: Final[List[float]] = [0.0, 0.0, 1.0, 1.0]
MAIN_VIEWPORT: Final[int] = -1

# TODO: Move here.
from habitat_hitl.core.ui_elements import UICanvasUpdate


# TODO: Deprecated legacy UI system.
@dataclass
class UIButton:
    """
    Networked UI button. Use RemoteClientState.ui_button_pressed() to retrieve state.
    """

    def __init__(self, button_id: str, text: str, enabled: bool):
        self.button_id = button_id
        self.text = text
        self.enabled = enabled


@dataclass
class UITextbox:
    """
    Networked UI textbox. Use RemoteClientState.get_textbox_content() to retrieve content.
    """

    def __init__(self, textbox_id: str, text: str, enabled: bool):
        self.textbox_id = textbox_id
        self.text = text
        self.enabled = enabled


class ClientMessageManager:
    r"""
    Extends gfx-replay keyframes to include server messages to be interpreted by the clients.
    Unlike keyframes, messages are client-specific.
    """
    _messages: List[Message]
    _users: Users

    def __init__(self, users: Users):
        self._users = users
        self.clear_messages()

    def any_message(self) -> bool:
        """
        Returns true if a message is ready to be sent for any user.
        """
        return any(len(message) > 0 for message in self._messages)

    def get_messages(self) -> List[Message]:
        r"""
        Get the messages to be communicated to each client.
        The list is indexed by user ID.
        """
        return self._messages

    def clear_messages(self) -> None:
        r"""Resets the messages."""
        self._messages = []
        for _ in range(self._users.max_user_count):
            self._messages.append({})

    def add_highlight(
        self,
        pos: List[float],
        radius: float,
        normal: List[float] = DEFAULT_NORMAL,
        billboard: bool = True,
        color: Optional[Union[mn.Color4, mn.Color3]] = None,
        destination_mask: Mask = Mask.ALL,
    ) -> None:
        r"""
        Draw a highlight circle around the specified position.
        """
        assert pos
        assert len(pos) == 3

        for user_index in self._users.indices(destination_mask):
            message = self._messages[user_index]
            if "circles" not in message:
                message["circles"] = []
            highlight_dict = {
                "t": [pos[0], pos[1], pos[2]],
                "r": radius,
                "n": normal,
            }
            if billboard:
                highlight_dict["b"] = 1
            if color is not None:

                def conv(channel):
                    # sloppy: using int 0-255 to reduce serialized data size
                    return int(channel * 255.0)

                alpha = 1.0 if isinstance(color, mn.Color3) else color.a
                highlight_dict["c"] = [
                    conv(color.r),
                    conv(color.g),
                    conv(color.b),
                    conv(alpha),
                ]
            message["circles"].append(highlight_dict)

    def add_line(
        self,
        a: List[float],
        b: List[float],
        from_color: Optional[Union[mn.Color4, mn.Color3]] = None,
        to_color: Optional[Union[mn.Color4, mn.Color3]] = None,
        destination_mask: Mask = Mask.ALL,
    ) -> None:
        r"""
        Draw a line from the two specified world positions.
        """
        assert len(a) == 3
        assert len(b) == 3

        for user_index in self._users.indices(destination_mask):
            message = self._messages[user_index]

            if "lines" not in message:
                message["lines"] = []
            lines_dict = {"a": [a[0], a[1], a[2]], "b": [b[0], b[1], b[2]]}

            if from_color is not None:

                def conv(channel):
                    # sloppy: using int 0-255 to reduce serialized data size
                    return int(channel * 255.0)

                alpha = (
                    1.0 if isinstance(from_color, mn.Color3) else from_color.a
                )
                lines_dict["c"] = [
                    conv(from_color.r),
                    conv(from_color.g),
                    conv(from_color.b),
                    conv(alpha),
                ]

            # TODO: Implement "to_color".

            message["lines"].append(lines_dict)

    def add_text(
        self, text: str, pos: list[float], destination_mask: Mask = Mask.ALL
    ):
        r"""
        Draw text at the specified screen coordinates.
        """
        for user_index in self._users.indices(destination_mask):
            message = self._messages[user_index]
            if len(text) == 0:
                return
            assert len(pos) == 2
            if "texts" not in message:
                message["texts"] = []
            message["texts"].append(
                {"text": text, "position": [pos[0], pos[1]]}
            )

    def show_modal_dialogue_box(
        self,
        title: str,
        text: str,
        buttons: List[UIButton],
        textbox: Optional[UITextbox] = None,
        destination_mask: Mask = Mask.ALL,
    ):
        r"""
        Show a modal dialog box with buttons.
        There can only be one modal dialog box at a time.
        """
        for user_index in self._users.indices(destination_mask):
            message = self._messages[user_index]

            message["dialog"] = {
                "title": title,
                "text": text,
                "buttons": [],
            }
            if textbox is not None:
                message["dialog"]["textbox"] = {
                    "id": textbox.textbox_id,
                    "text": textbox.text,
                    "enabled": textbox.enabled,
                }
            for button in buttons:
                message["dialog"]["buttons"].append(
                    {
                        "id": button.button_id,
                        "text": button.text,
                        "enabled": button.enabled,
                    }
                )

    def change_humanoid_position(
        self, pos: List[float], destination_mask: Mask = Mask.ALL
    ) -> None:
        r"""
        Change the position of the humanoid.
        Used to synchronize the humanoid position in the client when changing scene.
        """
        for user_index in self._users.indices(destination_mask):
            message = self._messages[user_index]
            message["teleportAvatarBasePosition"] = [pos[0], pos[1], pos[2]]

    def signal_scene_change(self, destination_mask: Mask = Mask.ALL) -> None:
        r"""
        Signals the client that the scene is being changed during this frame.
        """
        for user_index in self._users.indices(destination_mask):
            message = self._messages[user_index]
            message["sceneChanged"] = True

    def draw_object_outline(
        self,
        priority: int,
        color: List[float],
        line_width: float,
        object_ids: List[int],
        destination_mask: Mask = Mask.ALL,
    ) -> None:
        r"""
        Draw an outline around the specified objects.

        priority: Higher values are drawn in front of other outlines.
        color: Color of the outline.
        line_width: Width of the outline.
        object_ids: List of objects to outline.
        """
        for user_index in self._users.indices(destination_mask):
            message = self._messages[user_index]
            if "outlines" not in message:
                message["outlines"] = []
            message["outlines"].append(
                {
                    "priority": priority,
                    "color": color,
                    "width": line_width,
                    "objectIds": object_ids,
                }
            )

    def signal_app_ready(self, destination_mask: Mask = Mask.ALL):
        r"""
        See hitl_defaults.yaml wait_for_app_ready_signal documentation. Sloppy: this is a message to NetworkManager, not the client.
        """
        for user_index in self._users.indices(destination_mask):
            message = self._messages[user_index]
            message["isAppReady"] = True

    def set_server_keyframe_id(
        self, keyframe_id: int, destination_mask: Mask = Mask.ALL
    ):
        r"""
        Set the current keyframe ID.
        """
        for user_index in self._users.indices(destination_mask):
            message = self._messages[user_index]
            message["serverKeyframeId"] = keyframe_id

    def set_object_visibility_layer(
        self,
        object_id: int,
        layer_id: int = -1,
        destination_mask: Mask = Mask.ALL,
    ):
        r"""
        Set the visibility layer of the instance associated with specified habitat-sim objectId.
        The layer_id '-1' is the default layer and is visible to all viewports.
        There are 8 additional layers for controlling visibility (0 to 7).
        """
        assert layer_id >= -1
        assert layer_id < 8
        for user_index in self._users.indices(destination_mask):
            message = self._messages[user_index]
            object_properties = _obtain_object_properties(message, object_id)
            object_properties["layer"] = layer_id

    def set_viewport_properties(
        self,
        viewport_id: int,
        viewport_rect_xywh: List[float] = DEFAULT_VIEWPORT_SIZE,
        visible_layer_ids: Mask = Mask.ALL,
        destination_mask: Mask = Mask.ALL,
    ):
        r"""
        Set the properties of a viewport.
        Use viewport_id '-1' to edit the default viewport.

        viewport_id: Unique identifier of the viewport.
        viewport_rect_xywh: Viewport rect (x position, y position, width, height).
                            In window normalized coordinates, i.e. all values in range [0,1] relative to window size.
        visible_layer_ids: Visibility layers. Only objects assigned to these layers will be visible to this viewport.
        """
        layers = Users(8, activate_users=True)  # Maximum of 8 layers.
        for user_index in self._users.indices(destination_mask):
            message = self._messages[user_index]
            viewport_properties = _obtain_viewport_properties(
                message, viewport_id
            )
            # TODO: Use mask int instead of array
            viewport_properties["layers"] = []
            for layer in layers.indices(visible_layer_ids):
                viewport_properties["layers"].append(layer)
            viewport_properties["rect"] = viewport_rect_xywh

    def update_navmesh_triangles(
        self,
        triangle_vertices: List[List[float]],
        destination_mask: Mask = Mask.ALL,
    ):
        r"""
        Send a navmesh. triangle_vertices should be a list of vertices, 3 per triangle.
        Each vertex should be a 3-tuple or similar Iterable of floats.
        """
        assert len(triangle_vertices) > 0
        assert len(triangle_vertices) % 3 == 0
        assert len(triangle_vertices[0]) == 3

        for user_index in self._users.indices(destination_mask):
            message = self._messages[user_index]
            # flatten to a list of floats for more efficient serialization
            message["navmeshVertices"] = [
                component
                for sublist in triangle_vertices
                for component in sublist
            ]

    def update_camera_transform(
        self,
        cam_transform: mn.Matrix4,
        viewport_id: int = MAIN_VIEWPORT,
        destination_mask: Mask = Mask.ALL,
    ) -> None:
        r"""
        Update the camera transform of a viewport.
        Use the default 'viewport_id' 'MAIN_VIEWPORT' to update the main camera.
        Use 'set_viewport_properties' to create or update a viewport.
        """
        for user_index in self._users.indices(destination_mask):
            message = self._messages[user_index]
            pos = cam_transform.translation
            cam_rotation = mn.Quaternion.from_matrix(cam_transform.rotation())
            rot_vec = cam_rotation.vector
            rot = [
                cam_rotation.scalar,
                rot_vec[0],
                rot_vec[1],
                rot_vec[2],
            ]

            camera = _obtain_camera(message, viewport_id)
            camera["translation"] = [pos[0], pos[1], pos[2]]
            camera["rotation"] = [
                rot[0],
                rot[1],
                rot[2],
                rot[3],
            ]

    def update_ui_canvas(
        self,
        canvas_uid: str,
        canvas_update: UICanvasUpdate,
        destination_mask: Mask = Mask.ALL,
    ) -> None:
        r"""
        Update a UI canvas.
        """
        for user_index in self._users.indices(destination_mask):
            message = self._messages[user_index]
            if "uiUpdates" not in message:
                message["uiUpdates"] = {}
            message["uiUpdates"][canvas_uid] = asdict(canvas_update)


def _create_transform_dict(transform: mn.Matrix4) -> Dict[str, List[float]]:
    """Create a message dictionary from a transform."""
    p = transform.translation
    r = mn.Quaternion.from_matrix(transform.rotation())
    rv = r.vector
    return {
        "translation": [p[0], p[1], p[2]],
        "rotation": [r.scalar, rv[0], rv[1], rv[2]],
    }


def _obtain_object_properties(
    message: Message, object_id: int
) -> Dict[str, Any]:
    """Get or create the properties dict of an object_id."""
    if "objects" not in message:
        message["objects"] = {}
    if object_id not in message["objects"]:
        message["objects"][object_id] = {}
    return message["objects"][object_id]


def _obtain_viewport_properties(
    message: Message, viewport_id: int
) -> Dict[str, Any]:
    """Get or create the properties dict of a viewport."""
    if "viewports" not in message:
        message["viewports"] = {}
    if viewport_id not in message["viewports"]:
        message["viewports"][viewport_id] = {}
    return message["viewports"][viewport_id]


def _obtain_camera(message: Message, viewport_id: int) -> Dict[str, Any]:
    """Get or create the camera dict of a camera."""
    if "cameras" not in message:
        message["cameras"] = {}
    if viewport_id not in message["cameras"]:
        message["cameras"][viewport_id] = {}
    return message["cameras"][viewport_id]
