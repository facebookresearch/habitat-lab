#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from app_data import AppData
from app_state_base import AppStateBase
from app_states import create_app_state_start_session

from habitat_hitl.app_states.app_service import AppService


class AppStateLobby(AppStateBase):
    """
    Idle state.
    Ends when the target user count is reached.
    """

    def __init__(self, app_service: AppService, app_data: AppData):
        super().__init__(app_service, app_data)
        self._auto_save_keyframes = False

    def on_enter(self):
        super().on_enter()
        # Enable new connections
        # Sloppy: Create API
        self._app_service._remote_client_state._interprocess_record.enable_new_connections(
            True
        )

    def on_exit(self):
        super().on_exit()
        # Disable new connections
        # Sloppy: Create API
        self._app_service._remote_client_state._interprocess_record.enable_new_connections(
            False
        )

    def get_next_state(self) -> Optional[AppStateBase]:
        if (
            len(self._app_data.connected_users)
            == self._app_data.max_user_count
            and self._time_since_last_connection > 0.5
        ):
            return create_app_state_start_session(
                self._app_service, self._app_data
            )
        return None

    def sim_update(self, dt: float, post_sim_update_dict):
        missing_users = self._app_data.max_user_count - len(
            self._app_data.connected_users
        )
        s = "s" if missing_users > 1 else ""
        message = f"Waiting for {missing_users} participant{s} to join."
        self._status_message(message)
