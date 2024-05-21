#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Final, Optional

from app_data import AppData
from app_state_base import AppStateBase
from app_states import create_app_state_start_session

from habitat_hitl.app_states.app_service import AppService

# Delay to start the session after all users have connected.
# Occasionally, connection errors may occur rapidly after connecting, causing the session to start needlessly.
START_SESSION_DELAY: Final[float] = 0.5


class AppStateLobby(AppStateBase):
    """
    Idle state.
    Ends when the target user count is reached.
    """

    def __init__(self, app_service: AppService, app_data: AppData):
        super().__init__(app_service, app_data)
        self._save_keyframes = False

    def on_enter(self):
        super().on_enter()
        # Enable new connections
        # TODO: Create API in RemoteClientState
        self._app_service._remote_client_state._interprocess_record.enable_new_connections(
            True
        )

    def on_exit(self):
        super().on_exit()
        # Disable new connections
        # TODO: Create API in RemoteClientState
        self._app_service._remote_client_state._interprocess_record.enable_new_connections(
            False
        )

    def get_next_state(self) -> Optional[AppStateBase]:
        # If all users are connected, start the session.
        # NOTE: We wait START_SESSION_DELAY to mitigate early disconnects.
        if (
            len(self._app_data.connected_users)
            == self._app_data.max_user_count
            and self._time_since_last_connection > START_SESSION_DELAY
        ):
            return create_app_state_start_session(
                self._app_service, self._app_data
            )
        return None

    def sim_update(self, dt: float, post_sim_update_dict):
        # Show lobby status.
        missing_users = self._app_data.max_user_count - len(
            self._app_data.connected_users
        )
        if missing_users > 0:
            s = "s" if missing_users > 1 else ""
            message = f"Waiting for {missing_users} participant{s} to join."
            self._status_message(message)
        else:
            self._status_message("Loading...")
