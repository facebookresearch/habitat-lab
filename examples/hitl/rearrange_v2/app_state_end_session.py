#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from app_data import AppData
from app_state_base import AppStateBase
from app_states import create_app_state_reset
from session import Session
from util import get_top_down_view

from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.core.user_mask import Mask

# Duration of the end session message, before users are kicked.
SESSION_END_DELAY = 5.0


class AppStateEndSession(AppStateBase):
    """
    * Indicate users that the session is terminated.
    * Upload collected data.
    """

    def __init__(
        self, app_service: AppService, app_data: AppData, session: Session
    ):
        super().__init__(app_service, app_data)
        self._session = session
        self._elapsed_time = 0.0
        self._save_keyframes = False

        self._status = "Session ended."
        if len(session.error) > 0:
            self._status += f"\nError: {session.error}"

    def get_next_state(self) -> Optional[AppStateBase]:
        if self._elapsed_time > SESSION_END_DELAY:
            self._end_session()
            return create_app_state_reset(self._app_service, self._app_data)
        return None

    def sim_update(self, dt: float, post_sim_update_dict):
        # Top-down view.
        cam_matrix = get_top_down_view(self._app_service.sim)
        post_sim_update_dict["cam_transform"] = cam_matrix
        self._app_service._client_message_manager.update_camera_transform(
            cam_matrix, destination_mask=Mask.ALL
        )

        self._status_message(self._status)
        self._elapsed_time += dt

    def _end_session(self):
        # TODO: Data collection.
        pass
