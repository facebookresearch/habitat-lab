#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

from habitat_hitl.core.ui_elements import HorizontalAlignment

from app_data import AppData
from app_state_base import AppStateBase
from app_states import (
    create_app_state_cancel_session,
    create_app_state_rearrange,
)
from session import Session

from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.core.client_message_manager import UIButton
from habitat_hitl.core.key_mapping import KeyCode
from habitat_hitl.core.text_drawer import TextOnScreenAlignment
from habitat_hitl.core.user_mask import Mask

START_BUTTON_ID = "start"
START_SCREEN_TIMEOUT = 180.0
SKIP_START_SCREEN = True    # TODO: Start screen appears before episode loads.
FONT_SIZE_LARGE = 32
FONT_SIZE_SMALL = 24


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
        self._has_user_pressed_start_button: List[bool] = [
            False
        ] * self._app_data.max_user_count
        self._elapsed_time: float = 0.0
        self._timeout = False
        self._save_keyframes = True

    def get_next_state(self) -> Optional[AppStateBase]:
        if self._cancel:
            error = "Timeout" if self._timeout else "User disconnected"
            return create_app_state_cancel_session(
                self._app_service, self._app_data, self._session, error
            )

        # If all users pressed the "Start" button, begin the session.
        ready_to_start = True
        for user_ready in self._has_user_pressed_start_button:
            ready_to_start &= user_ready
        if ready_to_start or SKIP_START_SCREEN:
            return create_app_state_rearrange(
                self._app_service, self._app_data, self._session
            )

        return None

    def sim_update(self, dt: float, post_sim_update_dict):
        if SKIP_START_SCREEN:
            return

        # Time limit to start the experiment.
        self._elapsed_time += dt
        remaining_time = START_SCREEN_TIMEOUT - self._elapsed_time
        if remaining_time <= 0:
            self._cancel = True
            self._timeout = True
            return
        remaining_time_int = int(remaining_time)
        title = f"New Session (Expires in: {remaining_time_int}s)"

        for user_index in range(self._app_data.max_user_count):
            with self._app_service.ui_manager.update_canvas(
                "center", Mask.from_index(user_index)
            ) as ctx:
                ctx.canvas_properties(
                    padding=12, background_color=[0.3, 0.3, 0.3, 0.7]
                )

                ctx.label(
                    text=title,
                    font_size=FONT_SIZE_LARGE,
                    horizontal_alignment=HorizontalAlignment.CENTER,
                )

                ctx.separator()

                ctx.label(
                    text=f"Press start to being the experiment.\n\n",
                    font_size=FONT_SIZE_SMALL,
                    horizontal_alignment=HorizontalAlignment.CENTER,
                )

                if (
                    len(self._app_data.connected_users) > 1
                    and self._has_user_pressed_start_button[user_index]
                ):
                    button_text = "Waiting for other user..."
                else:
                    button_text = "OK"

                ctx.button(
                    uid=START_BUTTON_ID,
                    text=button_text,
                    enabled=not self._has_user_pressed_start_button[user_index],
                )
            self._has_user_pressed_start_button[
                user_index
            ] |= self._app_service.remote_client_state.ui_button_pressed(
                user_index, START_BUTTON_ID
            )

        # Server-only: Press numeric keys to start episode on behalf of users.
        if self._is_server_gui_enabled():
            server_message = "Press numeric keys to start on behalf of users."
            first_key = int(KeyCode.ONE)
            for user_index in range(len(self._has_user_pressed_start_button)):
                if self._app_service.gui_input.get_key_down(
                    KeyCode(first_key + user_index)
                ):
                    self._has_user_pressed_start_button[user_index] = True
                user_ready = self._has_user_pressed_start_button[user_index]
                server_message += f"\n[{user_index + 1}]: User {user_index}: {'Ready' if user_ready else 'Not ready'}."

            self._app_service.text_drawer.add_text(
                server_message,
                TextOnScreenAlignment.TOP_LEFT,
                text_delta_x=0,
                text_delta_y=-50,
                destination_mask=Mask.NONE,
            )
