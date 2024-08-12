#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from util import timestamp

from habitat_hitl.core.types import ConnectionRecord


@dataclass
class SessionRecord:
    episode_indices: List[int]
    session_error: str
    start_timestamp: int
    end_timestamp: int
    config: Dict[str, Any]
    frame_count: int
    connection_records: Dict[int, ConnectionRecord]


@dataclass
class UserRecord:
    user_index: int
    connection_record: ConnectionRecord


@dataclass
class EpisodeRecord:
    episode_index: int
    episode_id: str
    scene_id: str
    dataset: str
    user_index_to_agent_index_map: Dict[int, int]
    episode_info: Dict[str, Any]
    start_timestamp: int
    end_timestamp: int
    finished: bool
    task_percent_complete: float
    frame_count: int


@dataclass
class SessionOutput:
    session: SessionRecord
    users: List[UserRecord]
    episodes: List[EpisodeRecord]


@dataclass
class EpisodeOutput:
    session: SessionRecord
    users: List[UserRecord]
    episode: EpisodeRecord
    frames: List[Dict[str, Any]]


class SessionRecorder:
    def __init__(
        self,
        config: Dict[str, Any],
        connection_records: Dict[int, ConnectionRecord],
        episode_indices: List[int],
    ):
        time = timestamp()
        self.session_record = SessionRecord(
            episode_indices=episode_indices,
            session_error="",
            start_timestamp=time,
            end_timestamp=time,
            config=config,
            frame_count=0,
            connection_records=connection_records,
        )
        self.episode_records: List[EpisodeRecord] = []
        self.frames: List[List[Dict[str, Any]]] = []
        self.user_records: List[UserRecord] = []
        for user_index, connection_record in connection_records.items():
            self.user_records.append(
                UserRecord(
                    user_index=user_index, connection_record=connection_record
                )
            )

    def end_session(self, error: str):
        self.session_record.end_timestamp = timestamp()
        self.session_record.session_error = error

    def start_episode(
        self,
        episode_index: int,
        episode_id: str,
        scene_id: str,
        dataset: str,
        user_index_to_agent_index_map: Dict[int, int],
        episode_info: Dict[str, Any],
    ):
        time = timestamp()
        self.episode_records.append(
            EpisodeRecord(
                episode_index=episode_index,
                episode_id=episode_id,
                scene_id=scene_id,
                dataset=dataset,
                user_index_to_agent_index_map=user_index_to_agent_index_map,
                episode_info=episode_info,
                start_timestamp=time,
                end_timestamp=time,
                finished=False,
                task_percent_complete=0.0,
                frame_count=0,
            )
        )
        self.frames.append([])

    def end_episode(
        self,
        episode_finished: bool,
        task_percent_complete: float,
    ):
        assert len(self.episode_records) > 0
        episode = self.episode_records[-1]

        time = timestamp()
        self.session_record.end_timestamp = time
        episode.end_timestamp = time
        episode.finished = episode_finished
        episode.task_percent_complete = task_percent_complete

    def record_frame(
        self,
        frame_data: Dict[str, Any],
    ):
        assert len(self.episode_records) > 0
        episode_index = len(self.episode_records) - 1
        episode = self.episode_records[episode_index]

        time = timestamp()
        self.session_record.end_timestamp = time
        self.session_record.frame_count += 1
        episode.end_timestamp = time
        episode.frame_count += 1

        self.frames[episode_index].append(frame_data)

    def get_session_output(self) -> SessionOutput:
        return SessionOutput(
            self.session_record,
            self.user_records,
            self.episode_records,
        )

    def get_episode_outputs(self) -> List[EpisodeOutput]:
        output: List[EpisodeOutput] = []
        for i in range(len(self.episode_records)):
            output.append(
                EpisodeOutput(
                    self.session_record,
                    self.user_records,
                    self.episode_records[i],
                    self.frames[i],
                )
            )
        return output
