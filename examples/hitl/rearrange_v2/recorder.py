#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List

from util import timestamp

from habitat_hitl.core.types import ConnectionRecord


class SessionRecorder:
    def __init__(
        self,
        config: Dict[str, Any],
        connection_records: Dict[int, ConnectionRecord],
        episode_ids: List[str],
    ):
        self.data = {
            "episode_ids": episode_ids,
            "completed": False,
            "error": "",
            "start_timestamp": timestamp(),
            "end_timestamp": timestamp(),
            "config": config,
            "frame_count": 0,
            "users": [],
            "episodes": [],
        }

        for user_index in range(len(connection_records)):
            self.data["users"].append(
                {
                    "user_index": user_index,
                    # "agent_index": TODO: Only available during rearrange.
                    "connection_record": connection_records[user_index],
                }
            )

    def end_session(self, error: str):
        self.data["end_timestamp"] = timestamp()
        self.data["completed"] = True
        self.data["error"] = error

    def start_episode(
        self,
        episode_id: str,
        scene_id: str,
        dataset: str,
    ):
        self.data["episodes"].append(
            {
                "episode_id": episode_id,
                "scene_id": scene_id,
                "start_timestamp": timestamp(),
                "end_timestamp": timestamp(),
                "completed": False,
                "success": False,
                "frame_count": 0,
                "dataset": dataset,
                "frames": [],
            }
        )

    def end_episode(
        self,
        success: bool,
    ):
        self.data["episodes"][-1]["end_timestamp"] = timestamp()
        self.data["episodes"][-1]["success"] = success
        self.data["episodes"][-1]["completed"] = True

    def record_frame(
        self,
        frame_data: Dict[str, Any],
    ):
        self.data["end_timestamp"] = timestamp()
        self.data["frame_count"] += 1

        self.data["episodes"][-1]["end_timestamp"] = timestamp()
        self.data["episodes"][-1]["frame_count"] += 1
        self.data["episodes"][-1]["frames"].append(frame_data)
