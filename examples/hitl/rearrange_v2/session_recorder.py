#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional

from util import timestamp

from habitat_hitl.core.types import ConnectionRecord


class SessionRecorder:
    def __init__(
        self,
        config: Dict[str, Any],
        connection_records: Dict[int, ConnectionRecord],
        episode_indices: List[int],
    ):
        self.data = {
            "episode_indices": episode_indices,
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
        user_index_to_agent_index_map: Dict[int, int],
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
                "user_index_to_agent_index_map": user_index_to_agent_index_map,
                "frames": [],
            }
        )

    def end_episode(
        self,
        episode_finished: bool,
        task_percent_complete: float,
        task_explanation: Optional[str],
    ):
        self.data["episodes"][-1]["end_timestamp"] = timestamp()
        self.data["episodes"][-1]["finished"] = episode_finished
        self.data["episodes"][-1][
            "task_percent_complete"
        ] = task_percent_complete
        self.data["episodes"][-1]["task_explanation"] = task_explanation

    def record_frame(
        self,
        frame_data: Dict[str, Any],
    ):
        self.data["end_timestamp"] = timestamp()
        self.data["frame_count"] += 1

        self.data["episodes"][-1]["end_timestamp"] = timestamp()
        self.data["episodes"][-1]["frame_count"] += 1
        self.data["episodes"][-1]["frames"].append(frame_data)
