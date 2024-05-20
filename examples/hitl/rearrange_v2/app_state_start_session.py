#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

from app_data import AppData
from app_state_base import AppStateBase
from app_states import (
    create_app_state_cancel_session,
    create_app_state_load_episode,
)
from session import Session

from habitat_hitl.app_states.app_service import AppService


class AppStateStartSession(AppStateBase):
    def __init__(self, app_service: AppService, app_data: AppData):
        super().__init__(app_service, app_data)
        self._new_session: Optional[Session] = None
        self._save_keyframes = False

    def get_next_state(self) -> Optional[AppStateBase]:
        episode_ids = self._try_get_episode_ids()
        if episode_ids is not None:
            # Start the session.
            self._new_session = Session(
                self._app_service.config,
                list(episode_ids),
                dict(self._app_data.connected_users),
            )

            if self._cancel:
                return create_app_state_cancel_session(
                    self._app_service,
                    self._app_data,
                    self._new_session,
                    "User disconnected",
                )
            else:
                return create_app_state_load_episode(
                    self._app_service, self._app_data, self._new_session
                )
        else:
            # Create partial session record for data collection.
            self._new_session = Session(
                self._app_service.config,
                [],
                dict(self._app_data.connected_users),
            )
            return create_app_state_cancel_session(
                self._app_service,
                self._app_data,
                self._new_session,
                "Invalid session",
            )

    def _try_get_episode_ids(self) -> Optional[List[str]]:
        """
        Attempt to get episodes from client connection parameters.
        Episode IDs are indices within the episode sets.

        Format: {lower_bound_inclusive}-{upper_bound_exclusive} (e.g. "100-110").

        Returns None if the episode set cannot be resolved. This can happen in multiple cases:
        * 'episodes' field is missing from connection parameters.
        * Users are requesting different episodes in a multiplayer session, indicating a matching issue.
        * Invalid 'episodes' format.
        * Episode indices out of bounds.
        """
        data = self._app_data

        # Sanity checking.
        if len(data.connected_users) == 0:
            print("No user connected. Cancelling session.")
            return None
        connection_record = list(data.connected_users.values())[0]

        # Validate that episodes are selected.
        if "episodes" not in connection_record:
            print("Users did not request episodes. Cancelling session.")
            return None
        episodes_str = connection_record["episodes"]

        # Validate that all users are requesting the same episodes.
        for connection_record in data.connected_users.values():
            if connection_record["episodes"] != episodes_str:
                print(
                    "Users are requesting different episodes! Cancelling session."
                )
                return None

        # Validate that the episode set is not empty.
        if episodes_str is None or len(episodes_str) == 0:
            print("Users did not request episodes. Cancelling session.")
            return None

        # Format: {lower_bound}-{upper_bound} E.g. 100-110
        # Upper bound is exclusive.
        episode_range_str = episodes_str.split("-")
        if len(episode_range_str) != 2:
            print("Invalid episode range. Cancelling session.")
            return None

        # Validate that episodes are numeric.
        start_episode_id = (
            int(episode_range_str[0])
            if episode_range_str[0].isdecimal()
            else None
        )
        last_episode_id = (
            int(episode_range_str[1])
            if episode_range_str[0].isdecimal()
            else None
        )
        if (
            start_episode_id is None
            or last_episode_id is None
            or start_episode_id < 0
        ):
            print("Invalid episode names. Cancelling session.")
            return None

        total_episode_count = len(
            self._app_service.episode_helper._episode_iterator.episodes
        )

        # Validate episode range.
        if start_episode_id >= total_episode_count:
            print("Invalid episode names. Cancelling session.")
            return None

        if last_episode_id >= total_episode_count:
            last_episode_id = total_episode_count

        # If in decreasing order, swap.
        if start_episode_id > last_episode_id:
            temp = last_episode_id
            last_episode_id = start_episode_id
            start_episode_id = temp

        episode_ids: List[str] = []
        for episode_id in range(start_episode_id, last_episode_id):
            episode_ids.append(str(episode_id))

        return episode_ids
