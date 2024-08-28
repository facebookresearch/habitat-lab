#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List

from session_recorder import SessionRecorder

from habitat_hitl.core.types import ConnectionRecord


class Session:
    """
    Data for a single HITL session.
    A session is defined as a sequence of episodes done by a fixed set of users.
    """

    def __init__(
        self,
        config: Any,
        episode_indices: List[int],
        connection_records: Dict[int, ConnectionRecord],
    ):
        self.finished = False
        """Whether the session is finished."""

        self.episode_indices = episode_indices
        """List of episode indices within the session."""

        self.current_episode_index = 0
        """
        Current episode index within the episode set.

        If there are `1000` episodes, `current_episode_index` would be a value between `0` and `999` inclusively.
        """

        self.next_session_episode = 0
        """
        Next index of the `episode_indices` list (element index, not episode index).

        If `episode_indices` contains the values `10`, `20` and `30`, `next_session_episode` would be either `0`, `1`, `2` or `3`.
        """

        self.connection_records = connection_records
        """Connection records of each user."""

        self.session_recorder = SessionRecorder(
            config, connection_records, episode_indices
        )
        """Utility for recording the session."""

        self.error = ""
        """Field that contains the display error that caused session termination."""
