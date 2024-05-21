#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.core.client_message_manager import UIButton, UITextbox
from habitat_hitl.core.event import Event
from habitat_hitl.core.user_mask import Mask


@dataclass
class ErrorReport:
    """
    Error reported by a user.
    """

    user_message: str


@dataclass
class FormData:
    """
    Data contained within the end episode form.
    """

    app_service: AppService
    user_index: int
    current_state: Optional[BaseFormState]

    error_report_text: str

    on_cancel: Event
    on_episode_success: Event
    on_error_reported: Event


class BaseFormState:
    """
    Base state for a GUI form.
    """

    def __init__(self, data: FormData):
        self._data = data

    def step(self):
        pass

    def change_state(self, next_state: Optional[BaseFormState]):
        self._data.current_state = next_state

    def cancel(self):
        self._data.on_cancel.invoke(None)
        self.change_state(None)


class EndEpisodeFormState(BaseFormState):
    """
    End episode form.
    User is presented with the following options:
    * End episode
    * Report error
    * Cancel
    """

    def __init__(self, data: FormData):
        self._data = data

    def step(self):
        app_service = self._data.app_service
        user_index = self._data.user_index

        id_cancel = "cancel"
        id_success = "success"
        id_failure = "failure"
        buttons: List[UIButton] = [
            UIButton(id_cancel, "Cancel", enabled=True),
            UIButton(id_success, "Yes", enabled=True),
            UIButton(id_failure, "Report Error", enabled=True),
        ]
        app_service.client_message_manager.show_modal_dialogue_box(
            "End Task",
            "Was the task completed successfully?",
            buttons,
            destination_mask=Mask.from_index(user_index),
        )
        client_state = app_service.remote_client_state

        # If cancel button is clicked.
        if client_state.ui_button_pressed(user_index, id_cancel):
            self.cancel()

        # If episode finished button is clicked.
        if client_state.ui_button_pressed(user_index, id_success):
            self._data.on_episode_success.invoke(None)
            self.change_state(EpisodeSuccessFormState(self._data))

        # If report error button is clicked.
        if client_state.ui_button_pressed(user_index, id_failure):
            self.change_state(ErrorReportFormState(self._data))


class EpisodeSuccessFormState(BaseFormState):
    """
    Episode success form.
    User can cancel at any time.
    """

    def __init__(self, data: FormData):
        self._data = data

    def step(self):
        app_service = self._data.app_service
        user_index = self._data.user_index

        id_cancel = "cancel"
        buttons: List[UIButton] = [
            UIButton(id_cancel, "Cancel", enabled=True),
        ]
        app_service.client_message_manager.show_modal_dialogue_box(
            "Task Done",
            "Waiting for the other participant to finish...",
            buttons,
            destination_mask=Mask.from_index(user_index),
        )
        client_state = app_service.remote_client_state

        # If cancel button is clicked.
        if client_state.ui_button_pressed(user_index, id_cancel):
            self.cancel()


class ErrorReportFormState(BaseFormState):
    """
    Episode success form.
    User can cancel at any time.
    """

    def __init__(self, data: FormData):
        self._data = data

    def step(self):
        app_service = self._data.app_service
        user_index = self._data.user_index

        id_cancel = "cancel"
        id_report_error = "report_error"
        id_textbox = "report_text"
        buttons: List[UIButton] = [
            UIButton(id_cancel, "Cancel", enabled=True),
            UIButton(id_report_error, "Report Error", enabled=True),
        ]
        textbox = UITextbox(
            id_textbox, self._data.error_report_text, enabled=True
        )
        app_service.client_message_manager.show_modal_dialogue_box(
            "Report Error",
            "Write a short description of the problem.\nFor example: 'Could not find the phone'.",
            buttons,
            textbox=textbox,
            destination_mask=Mask.from_index(user_index),
        )
        client_state = app_service.remote_client_state

        # Read textbox content.
        self._data.error_report_text = client_state.get_textbox_content(
            user_index, id_textbox
        )

        # If cancel button is clicked.
        if client_state.ui_button_pressed(user_index, id_cancel):
            self.cancel()

        # If report error button is clicked.
        if client_state.ui_button_pressed(user_index, id_report_error):
            self._data.on_error_reported.invoke(
                ErrorReport(user_message=self._data.error_report_text)
            )
            self.change_state(ErrorReportedFormState(self._data))


class ErrorReportedFormState(BaseFormState):
    """
    Episode success form.
    User can cancel at any time.
    """

    def __init__(self, data: FormData):
        self._data = data

    def step(self):
        app_service = self._data.app_service
        user_index = self._data.user_index

        id_cancel = "cancel"
        buttons: List[UIButton] = [
            UIButton(id_cancel, "Cancel", enabled=True),
        ]
        id_textbox = "report_text_confirmation"
        textbox_report_confirmation = UITextbox(
            id_textbox, self._data.error_report_text, enabled=False
        )
        app_service.client_message_manager.show_modal_dialogue_box(
            "Error Reported",
            "Waiting for the other participant to finish...",
            buttons,
            textbox=textbox_report_confirmation,
            destination_mask=Mask.from_index(user_index),
        )

        # If cancel button is clicked.
        client_state = app_service.remote_client_state
        if client_state.ui_button_pressed(user_index, id_cancel):
            self.cancel()


class EndEpisodeForm:
    """
    Modal dialog box containing a form to signal episode completion.
    User can either signal the episode as completed, report an error or cancel to return to previous activity.
    The form is a state machine, each page being a state.
    """

    def __init__(
        self,
        user_index: int,
        app_service: AppService,
    ):
        self._data = FormData(
            app_service=app_service,
            user_index=user_index,
            current_state=None,
            error_report_text="",
            on_cancel=Event(),
            on_episode_success=Event(),
            on_error_reported=Event(),
        )

    def show(self):
        self._data.current_state = EndEpisodeFormState(self._data)

    def hide(self):
        self._data.current_state = None

    def is_form_shown(self) -> bool:
        return self._data.current_state != None

    def step(self):
        if self._data.current_state is None:
            return
        self._data.current_state.step()

    @property
    def on_cancel(self) -> Event:
        return self._data.on_cancel

    @property
    def on_episode_success(self) -> Event:
        return self._data.on_episode_success

    @property
    def on_error_reported(self) -> Event:
        return self._data.on_error_reported
