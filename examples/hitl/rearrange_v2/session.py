#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List

from recorder import SessionRecorder

from habitat_hitl.core.types import ConnectionRecord


class Session:
    """
    RearrangeV2 session.
    """

    def __init__(
        self,
        config: Any,
        episode_ids: List[str],
        connection_records: Dict[int, ConnectionRecord],
    ):
        self.success = False
        self.episode_ids = episode_ids
        self.current_episode_index = 0
        self.connection_records = connection_records
        self.session_recorder = SessionRecorder(
            config, connection_records, episode_ids
        )
        self.status = ""  # Use this to display error status

        # Use the port as a discriminator for when there are multiple concurrent servers.
        output_folder_suffix = str(config.habitat_hitl.networking.port)
        self.output_folder = f"output_{output_folder_suffix}"
