#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List


class ClientMessageManager:
    r"""
    Extends gfx-replay keyframes to include server messages to be interpreted by the client.
    """
    _message: Dict = {}

    def get_message_dict(self) -> Dict:
        r"""
        Get the server message to be communicated to the client.
        Add a field to this dict to send a message to the client at the end of the frame.
        """
        return self._message

    def clear_message_dict(self) -> None:
        r"""
        Resets the message dict.
        """
        self._message = {}

    def add_highlight(self, pos: List[float], radius: float) -> None:
        r"""
        Draw a highlight circle around the specified position.
        """
        assert pos
        assert len(pos) == 3

        if "highlights" not in self._message:
            self._message["highlights"] = []
        self._message["highlights"].append(
            {"t": [pos[0], pos[1], pos[2]], "r": radius}
        )

    def add_line(self, a: List[float], b: List[float]) -> None:
        r"""
        Draw a line from the two specified world positions.
        """
        assert len(a) == 3
        assert len(b) == 3

        if "lines" not in self._message:
            self._message["lines"] = []
        self._message["lines"].append(
            {"a": [a[0], a[1], a[2]], "b": [b[0], b[1], b[2]]}
        )

    def add_text(self, text: str, pos: list[float]):
        r"""
        Draw text at the specified screen positions.
        """
        if len(text) == 0:
            return
        assert len(pos) == 2
        if "texts" not in self._message:
            self._message["texts"] = []
        self._message["texts"].append({"text": text, "position": [pos[0], pos[1]]})

    def change_humanoid_position(self, pos: List[float]) -> None:
        r"""
        Change the position of the humanoid.
        Used to synchronize the humanoid position in the client when changing scene.
        """
        self._message["teleportAvatarBasePosition"] = [pos[0], pos[1], pos[2]]

    def signal_scene_change(self) -> None:
        r"""
        Signals the client that the scene is being changed during this frame.
        """
        self._message["sceneChanged"] = True

    def signal_app_ready(self):
        r"""
        See hitl_defaults.yaml wait_for_app_ready_signal documentation.
        """
        self._message["isAppReady"] = True

    def set_server_keyframe_id(self, keyframe_id):
        self._message["serverKeyframeId"] = keyframe_id

    def update_navmesh_triangles(self, triangle_vertices):
        r"""
        Send a navmesh. triangle_vertices should be a list of vertices, 3 per triangle.
        Each vertex should be a 3-tuple or similar Iterable of floats.
        """
        assert len(triangle_vertices) > 0
        assert len(triangle_vertices) % 3 == 0
        assert len(triangle_vertices[0]) == 3
        # flatten to a list of floats for more efficient serialization
        self._message["navmeshVertices"] = [
            component for sublist in triangle_vertices for component in sublist
        ]

    def update_camera_transform(self, pos: List[float], rot_quat: List[float]) -> None:
        r"""
        Update the camera transform.
        """
        assert len(pos) == 3
        assert len(rot_quat) == 4

        self._message["camera"] = {}
        self._message["camera"]["translation"] = [pos[0], pos[1], pos[2]]
        self._message["camera"]["rotation"] = [rot_quat[0], rot_quat[1], rot_quat[2], rot_quat[3]]
