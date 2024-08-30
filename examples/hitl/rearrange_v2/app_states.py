#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Boilerplate code for creating states without circular dependencies.
"""

from app_data import AppData
from app_state_base import AppStateBase
from session import Session

from habitat_hitl.app_states.app_service import AppService


def create_app_state_reset(
    app_service: AppService, app_data: AppData
) -> AppStateBase:
    from app_state_reset import AppStateReset

    return AppStateReset(app_service, app_data)


def create_app_state_lobby(
    app_service: AppService, app_data: AppData
) -> AppStateBase:
    from app_state_lobby import AppStateLobby

    return AppStateLobby(app_service, app_data)


def create_app_state_start_session(
    app_service: AppService, app_data: AppData
) -> AppStateBase:
    from app_state_start_session import AppStateStartSession

    return AppStateStartSession(app_service, app_data)


def create_app_state_load_episode(
    app_service: AppService, app_data: AppData, session: Session
) -> AppStateBase:
    from app_state_load_episode import AppStateLoadEpisode

    return AppStateLoadEpisode(app_service, app_data, session)


def create_app_state_start_screen(
    app_service: AppService, app_data: AppData, session: Session
) -> AppStateBase:
    from app_state_start_screen import AppStateStartScreen

    return AppStateStartScreen(app_service, app_data, session)


def create_app_state_rearrange(
    app_service: AppService, app_data: AppData, session: Session
) -> AppStateBase:
    from rearrange_v2 import AppStateRearrangeV2  # type: ignore

    return AppStateRearrangeV2(app_service, app_data, session)


def create_app_state_feedback(
    app_service: AppService,
    app_data: AppData,
    session: Session,
    success: float,
    feedback: str,
) -> AppStateBase:
    from app_state_feedback import AppStateFeedback

    return AppStateFeedback(app_service, app_data, session, success, feedback)


def create_app_state_end_session(
    app_service: AppService, app_data: AppData, session: Session
) -> AppStateBase:
    from app_state_end_session import AppStateEndSession

    return AppStateEndSession(app_service, app_data, session)


def create_app_state_cancel_session(
    app_service: AppService, app_data: AppData, session: Session, error: str
) -> AppStateBase:
    from app_state_end_session import AppStateEndSession

    session.error = error
    return AppStateEndSession(app_service, app_data, session)
