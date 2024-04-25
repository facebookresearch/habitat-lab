#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from app_data import AppData
from app_state_base import AppStateBase
from app_states import (
    create_app_state_end_session,
    create_app_state_start_screen,
)
from session import Session
from util import get_top_down_view

from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.core.user_mask import Mask


class AppStateLoadEpisode(AppStateBase):
    """
    Load an episode.
    A loading screen is displaying while the content loads.
    * If no episode set is selected, create a new episode set from connection record.
    * If a next episode exists, fires up RearrangeV2.
    * If all episodes are done, end session.
    Cancellable.
    """

    def __init__(
        self, app_service: AppService, app_data: AppData, session: Session
    ):
        super().__init__(app_service, app_data)
        self._session = session
        self._loading = True
        self._session_ended = False
        self._frame_number = 0
        self._auto_save_keyframes = False

    def get_next_state(self) -> Optional[AppStateBase]:
        if self._cancel:
            self._session.status = "User disconnected."
            return create_app_state_end_session(
                self._app_service, self._app_data, self._session
            )
        if self._session_ended:
            return create_app_state_end_session(
                self._app_service, self._app_data, self._session
            )
        # When all clients finish loading, show the start screen.
        if not self._loading:
            return create_app_state_start_screen(
                self._app_service, self._app_data, self._session
            )
        return None

    def sim_update(self, dt: float, post_sim_update_dict):
        self._status_message("Loading...")

        # HACK: Skip a frame so that the status message reaches the client before the server blocks.
        # TODO: Clean this up.
        if self._frame_number == 1:
            self._increment_episode()
            self._auto_save_keyframes = True  # HACK
        elif self._frame_number > 1:
            # Top-down view.
            cam_matrix = get_top_down_view(self._app_service.sim)
            post_sim_update_dict["cam_transform"] = cam_matrix
            self._app_service._client_message_manager.update_camera_transform(
                cam_matrix, destination_mask=Mask.ALL
            )

        # HACK: Sample periodically. Find a way to synchronize states properly.
        # Periodically check if clients are loading.
        if self._frame_number > 0 and self._frame_number % 20 == 0:
            any_client_loading = False
            for user_index in range(self._app_data.max_user_count):
                if self._app_service.remote_client_state._client_loading[
                    user_index
                ]:
                    any_client_loading = True
                    break
            if not any_client_loading:
                self._loading = False

        self._frame_number += 1

    def _increment_episode(self):
        data = self._app_data
        assert data.episode_ids is not None
        if data.current_episode_index < len(data.episode_ids):
            self._set_episode(data.current_episode_index)
            data.current_episode_index += 1
        else:
            self._session_ended = True

    def _set_episode(self, episode_index: int):
        data = self._app_data

        # Set the ID of the next episode to play in lab.
        next_episode_id = data.episode_ids[episode_index]
        self._app_service.episode_helper.set_next_episode_by_id(
            next_episode_id
        )

        # Once an episode ID has been set, lab needs to be reset to load the episode.
        self._app_service.end_episode(do_reset=True)

        # Signal the clients that the scene has changed.
        client_message_manager = self._app_service.client_message_manager
        if client_message_manager:
            client_message_manager.signal_scene_change(Mask.ALL)

        # Insert a keyframe to force clients to load immediately.
        self._app_service.sim.gfx_replay_manager.save_keyframe()

        # TODO: Timeout
