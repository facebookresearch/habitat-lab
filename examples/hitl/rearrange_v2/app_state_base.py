#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Optional

from app_data import AppData

from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.app_states.app_state_abc import AppState
from habitat_hitl.core.text_drawer import TextOnScreenAlignment
from habitat_hitl.core.user_mask import Mask


class AppStateBase(AppState):
    def __init__(
        self,
        app_service: AppService,
        app_data: AppData,
    ):
        self._app_service = app_service
        self._app_data = app_data
        self._cancel = False
        self._time_since_last_connection = 0
        self._auto_save_keyframes = True

    def on_enter(self):
        print(f"Entering state: {type(self)}")

    def on_exit(self):
        print(f"Exiting state: {type(self)}")

    def try_cancel(self):
        self._cancel = True

    def get_next_state(self) -> Optional[AppStateBase]:
        pass

    def on_environment_reset(self, episode_recorder_dict):
        pass

    def sim_update(self, dt: float, post_sim_update_dict):
        pass

    def record_state(self):
        pass

    def _status_message(self, message: str) -> None:
        """Send a message to all users."""
        if len(message) > 0:
            self._app_service.text_drawer.add_text(
                message,
                TextOnScreenAlignment.TOP_CENTER,
                text_delta_x=-280,
                text_delta_y=-50,
                destination_mask=Mask.ALL,
            )

    def _kick_all_users(self) -> None:
        "Kick all users."
        self._app_service.remote_client_state.kick(Mask.ALL)
