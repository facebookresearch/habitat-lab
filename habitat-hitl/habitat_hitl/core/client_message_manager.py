#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Union

import magnum as mn


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

    def add_highlight(
        self,
        pos: List[float],
        radius: float,
        billboard: bool = True,
        color: Optional[Union[mn.Color4, mn.Color3]] = None,
    ) -> None:
        r"""
        Draw a highlight circle around the specified position.
        """
        assert pos
        assert len(pos) == 3

        if "highlights" not in self._message:
            self._message["highlights"] = []
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
        self._message["highlights"].append(highlight_dict)

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
        See hitl_defaults.yaml wait_for_app_ready_signal documentation. Sloppy: this is a message to NetworkManager, not the client.
        """
        self._message["isAppReady"] = True

    def signal_kick_client(self, connection_id):
        r"""
        Signal NetworkManager to kick a client identified by connection_id. See also RemoteClientState.get_new_connection_records()[i]["connectionId"]. Sloppy: this is a message to NetworkManager, not the client.
        """
        self._message["kickClient"] = connection_id

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

    def update_camera_transform(self, cam_transform: mn.Matrix4) -> None:
        r"""
        Update the main camera transform.
        """
        pos = cam_transform.translation
        cam_rotation = mn.Quaternion.from_matrix(cam_transform.rotation())
        rot_vec = cam_rotation.vector
        rot = [
            cam_rotation.scalar,
            rot_vec[0],
            rot_vec[1],
            rot_vec[2],
        ]

        self._message["camera"] = {}
        self._message["camera"]["translation"] = [pos[0], pos[1], pos[2]]
        self._message["camera"]["rotation"] = [
            rot[0],
            rot[1],
            rot[2],
            rot[3],
        ]
