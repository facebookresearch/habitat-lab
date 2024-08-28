#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from app_data import AppData
from app_state_base import AppStateBase
from app_states import create_app_state_reset
from util import get_empty_view

from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.app_states.app_state_abc import AppState
from habitat_hitl.core.types import ConnectionRecord, DisconnectionRecord


class StateMachine(AppState):
    """
    RearrangeV2 state machine.
    It is itself an AppState containing sub-states.
    """

    def __init__(
        self,
        app_service: AppService,
    ):
        self._app_service = app_service
        self._app_data = AppData(
            app_service.hitl_config.networking.max_client_count
        )
        self._app_state: AppStateBase = create_app_state_reset(
            app_service, self._app_data
        )
        self._empty_view_matrix = get_empty_view(self._app_service.sim)

        if app_service.hitl_config.networking.enable:
            app_service.remote_client_state.on_client_connected.registerCallback(
                self._on_client_connected
            )
            app_service.remote_client_state.on_client_disconnected.registerCallback(
                self._on_client_disconnected
            )

    def _on_client_connected(self, connection: ConnectionRecord):
        user_index = connection["userIndex"]
        if user_index in self._app_data.connected_users:
            raise RuntimeError(
                f"User index {user_index} already connected! Aborting."
            )
        self._app_data.connected_users[connection["userIndex"]] = connection
        self._app_state._time_since_last_connection = 0.0
        self._app_service.users.activate_user(user_index)

    def _on_client_disconnected(self, disconnection: DisconnectionRecord):
        user_index = disconnection["userIndex"]
        if user_index not in self._app_data.connected_users:
            # TODO: Investigate why clients sometimes keep connecting/disconnecting.
            print(f"User index {user_index} already disconnected!")
            # raise RuntimeError(f"User index {user_index} already disconnected! Aborting.")
        else:
            del self._app_data.connected_users[user_index]

        self._app_service.users.deactivate_user(user_index)

        # If a user has disconnected, send a cancellation signal to the current state.
        self._app_state.try_cancel()

    def on_environment_reset(self, episode_recorder_dict):
        self._app_state.on_environment_reset(episode_recorder_dict)

    def sim_update(self, dt: float, post_sim_update_dict):
        self._app_state._time_since_last_connection += dt
        post_sim_update_dict["cam_transform"] = self._empty_view_matrix
        self._app_state.sim_update(dt, post_sim_update_dict)

        next_state = self._app_state.get_next_state()
        if next_state is not None:
            self._app_state.on_exit()
            self._app_service.ui_manager.reset()
            self._app_state = next_state
            self._app_state.on_enter()

        if self._app_state._save_keyframes == True:
            self._app_service.sim.gfx_replay_manager.save_keyframe()

    def record_state(self):
        pass  # Unused override.
