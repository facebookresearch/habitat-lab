#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List

from session_recorder import SessionRecorder

from habitat_hitl.core.types import ConnectionRecord


class Session:
    """
    Data for a single RearrangeV2 session.
    A session is defined as a sequence of episodes done by a fixed set of users.
    """

    def __init__(
        self,
        config: Any,
        episode_indices: List[int],
        connection_records: Dict[int, ConnectionRecord],
    ):
        self.success = False
        self.episode_indices = episode_indices
        self.current_episode_index = 0
        self.connection_records = connection_records

        self.session_recorder = SessionRecorder(
            config, connection_records, episode_indices
        )

        self.error = ""  # Use this to display error that causes termination
