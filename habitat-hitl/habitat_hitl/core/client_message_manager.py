#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Final, List, Optional, Union

import magnum as mn

from habitat_hitl.core.user_mask import Mask, Users

DEFAULT_NORMAL: Final[List[float]] = [0.0, 1.0, 0.0]


class ClientMessageManager:
    r"""
    Extends gfx-replay keyframes to include server messages to be interpreted by the clients.
    Unlike keyframes, messages are client-specific.
    """
    Message = Dict[str, Any]
    _messages: List[Message]
    _users: Users

    def __init__(self, users: Users):
        self._users = users
        self.clear_messages()

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

    def signal_app_ready(self, destination_mask: Mask = Mask.ALL):
        r"""
        See hitl_defaults.yaml wait_for_app_ready_signal documentation. Sloppy: this is a message to NetworkManager, not the client.
        """
        for user_index in self._users.indices(destination_mask):
            message = self._messages[user_index]
            message["isAppReady"] = True

    def signal_kick_client(
        self, connection_id: int, destination_mask: Mask = Mask.ALL
    ):
        r"""
        Signal NetworkManager to kick a client identified by connection_id. See also RemoteClientState.get_new_connection_records()[i]["connectionId"]. Sloppy: this is a message to NetworkManager, not the client.
        """
        for user_index in self._users.indices(destination_mask):
            message = self._messages[user_index]
            message["kickClient"] = connection_id

    def set_server_keyframe_id(
        self, keyframe_id: int, destination_mask: Mask = Mask.ALL
    ):
        r"""
        Set the current keyframe ID.
        """
        for user_index in self._users.indices(destination_mask):
            message = self._messages[user_index]
            message["serverKeyframeId"] = keyframe_id

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
        self, cam_transform: mn.Matrix4, destination_mask: Mask = Mask.ALL
    ) -> None:
        r"""
        Update the main camera transform.
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

            message["camera"] = {}
            message["camera"]["translation"] = [pos[0], pos[1], pos[2]]
            message["camera"]["rotation"] = [
                rot[0],
                rot[1],
                rot[2],
                rot[3],
            ]
