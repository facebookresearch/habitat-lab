#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Final, List, Optional

from app_data import AppData
from app_state_base import AppStateBase
from app_states import (
    create_app_state_cancel_session,
    create_app_state_load_episode,
)
from session import Session
from util import get_top_down_view

from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.core.key_mapping import KeyCode
from habitat_hitl.core.text_drawer import TextOnScreenAlignment
from habitat_hitl.core.ui_elements import HorizontalAlignment
from habitat_hitl.core.user_mask import Mask

OK_BUTTON_ID = "feedback_ok"
SCREEN_TIMEOUT = 30.0

FONT_SIZE_LARGE: Final[int] = 32
FONT_SIZE_SMALL: Final[int] = 24


class AppStateFeedback(AppStateBase):
    """
    Screen that shows task feedback for the recently finished episode.
    Cancellable.
    """

    def __init__(
        self,
        app_service: AppService,
        app_data: AppData,
        session: Session,
        success: float,
        feedback: str,
    ):
        super().__init__(app_service, app_data)
        self._session = session
        self._has_user_pressed_ok_button: List[bool] = [
            False
        ] * self._app_data.max_user_count
        self._elapsed_time: float = 0.0
        self._timeout = False
        self._save_keyframes = True
        self._cam_matrix = get_top_down_view(self._app_service.sim)
        self._success = success
        self._feedback = feedback

    def get_next_state(self) -> Optional[AppStateBase]:
        if self._cancel:
            error = "Timeout" if self._timeout else "User disconnected"
            return create_app_state_cancel_session(
                self._app_service, self._app_data, self._session, error
            )

        # If all users pressed the "Start" button, begin the session.
        ready_to_start = True
        for user_ready in self._has_user_pressed_ok_button:
            ready_to_start &= user_ready
        if ready_to_start or self._timeout:
            return create_app_state_load_episode(
                self._app_service, self._app_data, self._session
            )

        return None

    def sim_update(self, dt: float, post_sim_update_dict):
        # Top-down view.
        # TODO: Highlight objects and goals.
        cam_matrix = self._cam_matrix
        post_sim_update_dict["cam_transform"] = cam_matrix
        self._app_service._client_message_manager.update_camera_transform(
            cam_matrix, destination_mask=Mask.ALL
        )

        # Time limit to assess feedback.
        self._elapsed_time += dt
        remaining_time = SCREEN_TIMEOUT - self._elapsed_time
        if remaining_time <= 0:
            self._timeout = True
            return

        success = self._success
        if success > 0.99:
            title = "Task Success"
            content = "The task was completed successfully."
        else:
            title = f"Task Failure ({success:.0%} Success)"
            content = self._feedback

        for user_index in range(self._app_data.max_user_count):
            with self._app_service.ui_manager.update_canvas(
                "top_left", Mask.from_index(user_index)
            ) as ctx:
                ctx.canvas_properties(
                    padding=12, background_color=[0.3, 0.3, 0.3, 0.7]
                )

                ctx.label(
                    text=title,
                    font_size=FONT_SIZE_LARGE,
                    horizontal_alignment=HorizontalAlignment.LEFT,
                )

                ctx.separator()

                ctx.label(
                    text=f"{content}\n\n",
                    font_size=FONT_SIZE_SMALL,
                    horizontal_alignment=HorizontalAlignment.LEFT,
                )

                if (
                    len(self._app_data.connected_users) > 1
                    and self._has_user_pressed_ok_button[user_index]
                ):
                    button_text = "Waiting for other user..."
                else:
                    button_text = "OK"

                ctx.button(
                    uid="feedback_button",
                    text=button_text,
                    enabled=not self._has_user_pressed_ok_button[user_index],
                )
            self._has_user_pressed_ok_button[
                user_index
            ] |= self._app_service.remote_client_state.ui_button_pressed(
                user_index, "feedback_button"
            )

        # Server-only: Press numeric keys to assess feedback on behalf of users.
        if self._is_server_gui_enabled():
            server_message = "Press numeric keys to assess on behalf of users."
            first_key = int(KeyCode.ONE)
            for user_index in range(len(self._has_user_pressed_ok_button)):
                if self._app_service.gui_input.get_key_down(
                    KeyCode(first_key + user_index)
                ):
                    self._has_user_pressed_ok_button[user_index] = True
                user_ready = self._has_user_pressed_ok_button[user_index]
                server_message += f"\n[{user_index + 1}]: User {user_index}: {'Ready' if user_ready else 'Not ready'}."

            self._app_service.text_drawer.add_text(
                server_message,
                TextOnScreenAlignment.TOP_LEFT,
                text_delta_x=0,
                text_delta_y=-50,
                destination_mask=Mask.NONE,
            )
