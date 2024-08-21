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
        episode_indices = self._try_get_episode_indices(
            data=self._app_data,
            total_episode_count=len(
                self._app_service.episode_helper._episode_iterator.episodes
            ),
        )
        if episode_indices is not None:
            # Start the session.
            self._new_session = Session(
                self._app_service.config,
                list(episode_indices),
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

    @staticmethod
    def _try_get_episode_indices(
        data: AppData, total_episode_count: int
    ) -> Optional[List[int]]:
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

        # Sanity checking.
        user_count = len(data.connected_users)
        if user_count == 0:
            print("No user connected. Cancelling session.")
            return None
        episodes: List[List[int]] = []

        # Validate that the episodes are integers.
        for user_index in range(user_count):
            episodes.append([])
            connection_record = list(data.connected_users.values())[user_index]

            # Validate that episodes are selected.
            if "episodes" not in connection_record:
                print("Users did not request episodes. Cancelling session.")
                return None
            episodes_str = connection_record["episodes"]

            # Validate that the parameter is a string.
            if episodes_str is None or not isinstance(episodes_str, str):
                print(
                    f"Episodes are supplied in an unexpected format: '{type(episodes_str)}'. Cancelling session."
                )
                return None

            # Parse episode string.
            episodes_split = episodes_str.split(",")

            for episode_str in episodes_split:
                try:
                    episodes[user_index].append(int(episode_str))
                except Exception:
                    print(
                        f"Episode index '{episode_str}' is not an integer. Cancelling session."
                    )
                    return None

            # Validate that the episode set is not empty.
            if len(episodes[0]) == 0:
                print("User did not request episodes. Cancelling session.")
                return None

            # Validate that all users are requesting the same episodes.
            if user_index > 0 and episodes[user_index] != episodes[0]:
                print(
                    "Users are requesting different episodes. Cancelling session."
                )
                return None

        # Validate that the episodes are within range.
        for episode in episodes[0]:
            if episode < 0 or episode >= total_episode_count:
                print(
                    f"Episode index '{episode}' is out of bounds. There are '{total_episode_count}' episodes in this dataset. Cancelling session."
                )
                return None

        return list(episodes[0])
