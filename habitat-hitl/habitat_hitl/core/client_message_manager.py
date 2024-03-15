#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Union
from habitat_hitl.core.user_mask import UserMask, UserMaskIterator

import magnum as mn


class ClientMessageManager:
    r"""
    Extends gfx-replay keyframes to include server messages to be interpreted by the clients.
    """
    # [0mdc/multiplayer] TODO: List of messages
    Message = Dict[str, Any]
    _messages: List[Message]

    def __init__(self, max_user_count: int):
        self._max_user_count = max_user_count
        self._user_mask_iter = UserMaskIterator(max_user_count)
        self.clear_messages()

    def get_messages(self) -> List[Message]:
        r"""
        Get the server message to be communicated to the client.
        Add a field to this dict to send a message to the client at the end of the frame.
        """
        return self._messages

    def clear_messages(self) -> None:
        r"""
        Resets the message dict.
        """
        self._messages = []
        for _ in range(self._max_user_count):
            self._messages.append({})

    def add_highlight(
        self,
        pos: List[float],
        radius: float,
        billboard: bool = True,
        color: Optional[Union[mn.Color4, mn.Color3]] = None,
        destination_mask: UserMask = UserMask.BROADCAST,
    ) -> None:
        r"""
        Draw a highlight circle around the specified position.
        """
        for user_index in self._user_mask_iter.user_indices(destination_mask):
            assert pos
            assert len(pos) == 3
            message = self._messages[user_index]

            if "highlights" not in message:
                message["highlights"] = []
            highlight_dict = {"t": [pos[0], pos[1], pos[2]], "r": radius}
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
            message["highlights"].append(highlight_dict)

    def add_text(self, text: str, pos: list[float], destination_mask: UserMask = UserMask.BROADCAST):
        r"""
        Draw text at the specified screen positions.
        """
        for user_index in self._user_mask_iter.user_indices(destination_mask):
            if len(text) == 0:
                return
            assert len(pos) == 2
            if "texts" not in self._messages[user_index]:
                self._messages[user_index]["texts"] = []
            self._messages[user_index]["texts"].append(
                {"text": text, "position": [pos[0], pos[1]]}
            )

    def change_humanoid_position(self, pos: List[float], destination_mask: UserMask = UserMask.BROADCAST) -> None:
        r"""
        Change the position of the humanoid.
        Used to synchronize the humanoid position in the client when changing scene.
        """
        for user_index in self._user_mask_iter.user_indices(destination_mask):
            self._messages[user_index]["teleportAvatarBasePosition"] = [pos[0], pos[1], pos[2]]

    def signal_scene_change(self, destination_mask: UserMask = UserMask.BROADCAST) -> None:
        r"""
        Signals the client that the scene is being changed during this frame.
        """
        for user_index in self._user_mask_iter.user_indices(destination_mask):
            self._messages[user_index]["sceneChanged"] = True

    def signal_app_ready(self, destination_mask: UserMask = UserMask.BROADCAST):
        r"""
        See hitl_defaults.yaml wait_for_app_ready_signal documentation. Sloppy: this is a message to NetworkManager, not the client.
        """
        for user_index in self._user_mask_iter.user_indices(destination_mask):
            self._messages[user_index]["isAppReady"] = True

    def signal_kick_client(self, connection_id: int, destination_mask: UserMask = UserMask.BROADCAST):
        r"""
        Signal NetworkManager to kick a client identified by connection_id. See also RemoteClientState.get_new_connection_records()[i]["connectionId"]. Sloppy: this is a message to NetworkManager, not the client.
        """
        for user_index in self._user_mask_iter.user_indices(destination_mask):
            self._messages[user_index]["kickClient"] = connection_id

    def set_server_keyframe_id(self, keyframe_id: int, destination_mask: UserMask = UserMask.BROADCAST):
        for user_index in self._user_mask_iter.user_indices(destination_mask):
            self._messages[user_index]["serverKeyframeId"] = keyframe_id

    def update_navmesh_triangles(self, triangle_vertices, destination_mask: UserMask = UserMask.BROADCAST):
        r"""
        Send a navmesh. triangle_vertices should be a list of vertices, 3 per triangle.
        Each vertex should be a 3-tuple or similar Iterable of floats.
        """
        for user_index in self._user_mask_iter.user_indices(destination_mask):
            assert len(triangle_vertices) > 0
            assert len(triangle_vertices) % 3 == 0
            assert len(triangle_vertices[0]) == 3
            # flatten to a list of floats for more efficient serialization
            self._messages[user_index]["navmeshVertices"] = [
                component for sublist in triangle_vertices for component in sublist
            ]

    def update_camera_transform(self, cam_transform: mn.Matrix4, destination_mask: UserMask = UserMask.BROADCAST) -> None:
        r"""
        Update the main camera transform.
        """
        for user_index in self._user_mask_iter.user_indices(destination_mask):
            pos = cam_transform.translation
            cam_rotation = mn.Quaternion.from_matrix(cam_transform.rotation())
            rot_vec = cam_rotation.vector
            rot = [
                cam_rotation.scalar,
                rot_vec[0],
                rot_vec[1],
                rot_vec[2],
            ]

            self._messages[user_index]["camera"] = {}
            self._messages[user_index]["camera"]["translation"] = [pos[0], pos[1], pos[2]]
            self._messages[user_index]["camera"]["rotation"] = [
                rot[0],
                rot[1],
                rot[2],
                rot[3],
            ]
