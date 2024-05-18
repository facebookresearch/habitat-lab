#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from app_data import AppData
from app_state_base import AppStateBase
from app_states import create_app_state_lobby

from habitat_hitl.app_states.app_service import AppService


class AppStateReset(AppStateBase):
    """
    Kick all users and restore state for a new session.
    """

    def __init__(self, app_service: AppService, app_data: AppData):
        super().__init__(app_service, app_data)
        self._save_keyframes = False

    def on_enter(self):
        super().on_enter()

        # Kick all users.
        self._kick_all_users()

    def get_next_state(self) -> Optional[AppStateBase]:
        # Wait for users to be kicked.
        if len(self._app_data.connected_users) == 0:
            return create_app_state_lobby(self._app_service, self._app_data)
        else:
            return None
