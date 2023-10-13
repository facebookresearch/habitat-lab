#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

class MessagingService():
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

    def add_highlight(self, pos: List[float]) -> None:
        r"""
        Draw a highlight circle around the specified position.
        """
        assert pos
        assert len(pos) == 3

        if not "highlights" in self._message:
            self._message["highlights"] = []
        self._message["highlights"].append({"t": [pos[0], pos[1], pos[2]]})

    def add_message_to_keyframe(self, keyframe_obj) -> None:
        r"""
        Adds the server message to the specified keyframe.
        Clears the server message.
        """
        keyframe_obj["message"] = self._message
        self._message = {}
