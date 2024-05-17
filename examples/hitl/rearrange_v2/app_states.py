#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Boilerplate code for creating states without circular dependencies.
"""

from app_data import AppData
from app_state_base import AppStateBase

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


def create_app_state_rearrange(
    app_service: AppService, app_data: AppData
) -> AppStateBase:
    from rearrange_v2 import AppStateRearrangeV2

    return AppStateRearrangeV2(app_service, app_data)
