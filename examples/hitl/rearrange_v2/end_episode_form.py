#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import List

from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.core.client_message_manager import UIButton, UITextbox
from habitat_hitl.core.user_mask import Mask


class EndEpisodeForm:
    class Result(Enum):
        CANCEL = 0
        PENDING = 1
        SUCCESS = 2
        FAILURE = 3

    class State(Enum):
        END_EPISODE_FORM = 0
        ERROR_REPORT_FORM = 1
        SUCCESS_MESSAGE = 2
        FAILURE_MESSAGE = 3

    def __init__(
        self,
        user_index: int,
        app_service: AppService,
    ):
        self._user_index = user_index
        self._app_service = app_service
        self._state: EndEpisodeForm.State = (
            EndEpisodeForm.State.END_EPISODE_FORM
        )
        self._error_report_text = ""

    def show(self) -> Result:
        if self._state == EndEpisodeForm.State.END_EPISODE_FORM:
            return self._show_end_episode_form()
        elif self._state == EndEpisodeForm.State.ERROR_REPORT_FORM:
            return self._show_error_report_form()
        elif self._state == EndEpisodeForm.State.SUCCESS_MESSAGE:
            return self._show_success_form()
        elif self._state == EndEpisodeForm.State.FAILURE_MESSAGE:
            return self._show_failure_form()
        return self._cancel()

    def _show_end_episode_form(self) -> Result:
        user_index = self._user_index
        id_cancel = "cancel"
        id_success = "success"
        id_failure = "failure"
        buttons: List[UIButton] = [
            UIButton(id_cancel, "Cancel", enabled=True),
            UIButton(id_success, "Yes", enabled=True),
            UIButton(id_failure, "Report Error", enabled=True),
        ]
        self._app_service.client_message_manager.show_modal_dialogue_box(
            "End Task",
            "Was the task completed successfully?",
            buttons,
            destination_mask=Mask.from_index(user_index),
        )
        client_state = self._app_service.remote_client_state

        # If cancel button is clicked.
        if client_state.ui_button_clicked(user_index, id_cancel):
            return self._cancel()

        # If success button is clicked.
        if client_state.ui_button_clicked(user_index, id_success):
            self._state = EndEpisodeForm.State.SUCCESS_MESSAGE

        # If error report button is clicked.
        if client_state.ui_button_clicked(user_index, id_failure):
            self._state = EndEpisodeForm.State.ERROR_REPORT_FORM

        return EndEpisodeForm.Result.PENDING

    def _show_error_report_form(self) -> Result:
        user_index = self._user_index
        id_cancel = "cancel"
        id_report_error = "report_error"
        id_textbox = "report_text"
        buttons: List[UIButton] = [
            UIButton(id_cancel, "Cancel", enabled=True),
            UIButton(id_report_error, "Report Error", enabled=True),
        ]
        textbox = UITextbox(id_textbox, self._error_report_text, enabled=True)
        self._app_service.client_message_manager.show_modal_dialogue_box(
            "Report Error",
            "Write a short description of the problem.\nFor example: 'Could not find the phone'.",
            buttons,
            textbox=textbox,
            destination_mask=Mask.from_index(user_index),
        )
        client_state = self._app_service.remote_client_state

        # Read textbox content.
        self._error_report_text = client_state.get_textbox_content(
            user_index, id_textbox
        )

        # If cancel button is clicked.
        if client_state.ui_button_clicked(user_index, id_cancel):
            return self._cancel()

        # If report error button is clicked.
        if client_state.ui_button_clicked(user_index, id_report_error):
            self._state = EndEpisodeForm.State.FAILURE_MESSAGE

        return EndEpisodeForm.Result.PENDING

    def _show_success_form(self) -> Result:
        user_index = self._user_index
        id_cancel = "cancel"
        buttons: List[UIButton] = [
            UIButton(id_cancel, "Cancel", enabled=True),
        ]
        self._app_service.client_message_manager.show_modal_dialogue_box(
            "Task Done",
            "Waiting for the other participant to finish...",
            buttons,
            destination_mask=Mask.from_index(user_index),
        )
        client_state = self._app_service.remote_client_state

        # If cancel button is clicked.
        if client_state.ui_button_clicked(user_index, id_cancel):
            return self._cancel()

        return EndEpisodeForm.Result.SUCCESS

    def _show_failure_form(self) -> Result:
        user_index = self._user_index
        id_cancel = "cancel"
        buttons: List[UIButton] = [
            UIButton(id_cancel, "Cancel", enabled=True),
        ]
        id_textbox = "report_text_confirmation"
        textbox_report_confirmation = UITextbox(
            id_textbox, self._error_report_text, enabled=False
        )
        self._app_service.client_message_manager.show_modal_dialogue_box(
            "Error Reported",
            "Waiting for the other participant to finish...",
            buttons,
            textbox=textbox_report_confirmation,
            destination_mask=Mask.from_index(user_index),
        )

        # If cancel button is clicked.
        client_state = self._app_service.remote_client_state
        if client_state.ui_button_clicked(user_index, id_cancel):
            return self._cancel()

        return EndEpisodeForm.Result.FAILURE

    def _cancel(self) -> Result:
        self._error_report_text = ""
        self._state = EndEpisodeForm.State.END_EPISODE_FORM
        return EndEpisodeForm.Result.CANCEL
