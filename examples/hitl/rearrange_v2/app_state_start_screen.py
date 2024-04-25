#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

from app_data import AppData
from app_state_base import AppStateBase
from app_states import create_app_state_end_session, create_app_state_rearrange
from session import Session
from util import get_top_down_view

from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.core.client_message_manager import UIButton
from habitat_hitl.core.key_mapping import KeyCode
from habitat_hitl.core.text_drawer import TextOnScreenAlignment
from habitat_hitl.core.user_mask import Mask

START_BUTTON_ID = "start"
START_SCREEN_TIMEOUT = 120.0
SKIP_START_SCREEN = False


class AppStateStartScreen(AppStateBase):
    """
    Start screen with a "Start" button that all users must press before starting the session.
    Cancellable.
    """

    def __init__(
        self, app_service: AppService, app_data: AppData, session: Session
    ):
        super().__init__(app_service, app_data)
        self._session = session
        self._ready_to_start: List[bool] = [
            False
        ] * self._app_data.max_user_count
        self._elapsed_time: float = 0.0
        self._timeout = False  # TODO: Error management
        self._save_keyframes = True

    def get_next_state(self) -> Optional[AppStateBase]:
        if self._cancel:
            self._session.error = (
                "Timeout" if self._timeout else "User disconnected"
            )
            return create_app_state_end_session(
                self._app_service, self._app_data, self._session
            )

        # If all users pressed the "Start" button, begin the session.
        ready_to_start = True
        for user_ready in self._ready_to_start:
            ready_to_start &= user_ready
        if ready_to_start or SKIP_START_SCREEN:
            return create_app_state_rearrange(
                self._app_service, self._app_data, self._session
            )

        return None

    def sim_update(self, dt: float, post_sim_update_dict):
        # Top-down view.
        cam_matrix = get_top_down_view(self._app_service.sim)
        post_sim_update_dict["cam_transform"] = cam_matrix
        self._app_service._client_message_manager.update_camera_transform(
            cam_matrix, destination_mask=Mask.ALL
        )

        # Time limit to start the experiment.
        self._elapsed_time += dt
        remaining_time = START_SCREEN_TIMEOUT - self._elapsed_time
        if remaining_time <= 0:
            self._cancel = True
            self._timeout = True
            return
        remaining_time_int = int(remaining_time)
        title = f"New Session (Expires in: {remaining_time_int}s)"

        # Show dialogue box with "Start" button.
        for user_index in range(self._app_data.max_user_count):
            button_pressed = (
                self._app_service.remote_client_state.ui_button_clicked(
                    user_index, START_BUTTON_ID
                )
            )
            self._ready_to_start[user_index] |= button_pressed

            if not self._ready_to_start[user_index]:
                self._app_service.client_message_manager.show_modal_dialogue_box(
                    title,
                    "Press 'Start' to begin the experiment.",
                    [UIButton(START_BUTTON_ID, "Start", True)],
                    Mask.from_index(user_index),
                )
            else:
                self._app_service.client_message_manager.show_modal_dialogue_box(
                    title,
                    "Waiting for other participants...",
                    [UIButton(START_BUTTON_ID, "Start", False)],
                    Mask.from_index(user_index),
                )

        # Server-only: Press numeric keys to press button on behalf of users.
        if not self._app_service.hitl_config.experimental.headless.do_headless:
            server_message = "Press numeric keys to start on behalf of users."
            first_key = int(KeyCode.ONE)
            for user_index in range(len(self._ready_to_start)):
                if self._app_service.gui_input.get_key_down(
                    KeyCode(first_key + user_index)
                ):
                    self._ready_to_start[user_index] = True
                user_ready = self._ready_to_start[user_index]
                server_message += f"\n[{user_index + 1}]: User {user_index}: {'Ready' if user_ready else 'Not ready'}."

            self._app_service.text_drawer.add_text(
                server_message,
                TextOnScreenAlignment.TOP_LEFT,
                text_delta_x=0,
                text_delta_y=-50,
                destination_mask=Mask.NONE,
            )
