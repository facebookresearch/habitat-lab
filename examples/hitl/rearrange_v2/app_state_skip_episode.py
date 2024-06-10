#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from app_data import AppData
from app_state_base import AppStateBase
from app_states import (
    create_app_state_cancel_session,
    create_app_state_load_episode,
)
from session import Session
from util import get_top_down_view

from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.core.user_mask import Mask

SKIP_EPISODE_MESSAGE_DURATION = 5.0


class AppStateSkipEpisode(AppStateBase):
    """
    Skip an episode.
    A message is displayed for 'SKIP_EPISODE_MESSAGE_DURATION' before resuming session.
    """

    def __init__(
        self,
        app_service: AppService,
        app_data: AppData,
        session: Session,
        message: str,
    ):
        super().__init__(app_service, app_data)
        self._session = session
        self._message = message
        self._timer = SKIP_EPISODE_MESSAGE_DURATION

    def get_next_state(self) -> Optional[AppStateBase]:
        if self._cancel:
            return create_app_state_cancel_session(
                self._app_service,
                self._app_data,
                self._session,
                "User disconnected.",
            )
        if self._timer < 0.0:
            return create_app_state_load_episode(
                self._app_service, self._app_data, self._session
            )
        return None

    def sim_update(self, dt: float, post_sim_update_dict):
        self._status_message(self._message)
        self._timer -= dt

        cam_matrix = get_top_down_view(self._app_service.sim)
        post_sim_update_dict["cam_transform"] = cam_matrix
        self._app_service._client_message_manager.update_camera_transform(
            cam_matrix, destination_mask=Mask.ALL
        )
