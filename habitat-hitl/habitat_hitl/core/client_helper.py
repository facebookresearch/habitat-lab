#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from datetime import datetime, timedelta
from typing import List, Optional

from habitat_hitl.core.average_helper import AverageHelper
from habitat_hitl.core.client_message_manager import ClientMessageManager
from habitat_hitl.core.types import ConnectionRecord, DisconnectionRecord
from habitat_hitl.core.user_mask import Mask, Users


class ClientHelper:
    """
    Tracks connected remote clients. Displays client latency and kicks idle clients.
    """

    def __init__(
        self,
        hitl_config,
        remote_client_state,
        client_message_manager: ClientMessageManager,
        users: Users,
    ):
        assert hitl_config.networking.enable
        self._remote_client_state = remote_client_state
        self._client_message_manager = client_message_manager
        self._users = users
        self._connected_users = Mask.NONE

        self._kick_active: bool = (
            hitl_config.networking.client_max_idle_duration != None
        )
        self._max_idle_duration: Optional[
            int
        ] = hitl_config.networking.client_max_idle_duration

        user_count = users.max_user_count
        self._show_idle_kick_warning: List[bool] = [False] * user_count
        self._last_activity: List[datetime] = [datetime.now()] * user_count
        self._display_latency_ms: List[Optional[float]] = [None] * user_count
        self._client_frame_latency_avg_helper: List[
            Optional[AverageHelper]
        ] = [None] * user_count
        self._frame_counter: List[int] = [0] * user_count

        remote_client_state.on_client_connected.registerCallback(
            self._on_client_connected
        )
        remote_client_state.on_client_disconnected.registerCallback(
            self._on_client_disconnected
        )

    def activate_users(self) -> None:
        """
        Reset idle timer for all users.
        """
        for user_index in range(self._users.max_user_count):
            self._show_idle_kick_warning[user_index] = False
            self._last_activity[user_index] = datetime.now()

    def _reset_user(self, user_index: int):
        self._show_idle_kick_warning[user_index] = False
        self._last_activity[user_index] = datetime.now()
        self._display_latency_ms[user_index] = None
        self._client_frame_latency_avg_helper[user_index] = AverageHelper(
            window_size=10, output_rate=10
        )
        self._frame_counter[user_index] = 0

    def _on_client_connected(self, connection: ConnectionRecord):
        user_index = connection["userIndex"]
        self._connected_users |= Mask.from_index(user_index)
        self._reset_user(user_index)

    def _on_client_disconnected(self, disconnection: DisconnectionRecord):
        user_index = disconnection["userIndex"]
        self._connected_users &= ~Mask.from_index(user_index)
        self._reset_user(user_index)

    def display_latency_ms(self, user_index: int) -> Optional[float]:
        """Returns the display latency."""
        return self._display_latency_ms[user_index]

    def do_show_idle_kick_warning(self, user_index: int) -> Optional[bool]:
        """Indicates that the user should be warned that they will be kicked imminently."""
        return self._show_idle_kick_warning[user_index]

    def get_idle_time(self, user_index: int) -> int:
        """Returns the current idle time."""
        if not self._kick_active:
            return 0
        now = datetime.now()
        last_activity = self._last_activity[user_index]
        span = now - last_activity
        return int(span.total_seconds())

    def get_remaining_idle_time(self, user_index: int) -> int:
        """Returns the remaining idle time before kicking."""
        if not self._kick_active:
            return 0
        return int(self._max_idle_duration - self.get_idle_time(user_index))

    def _update_idle_kick(
        self, user_index: int, is_user_idle_this_frame: bool
    ) -> None:
        """Tracks whether the user is idle. After some time, they will be kicked."""

        if not self._kick_active or user_index not in self._users.indices(
            self._connected_users
        ):
            return

        self._show_idle_kick_warning[user_index] = False

        now = datetime.now()
        if not is_user_idle_this_frame:
            self._last_activity[user_index] = now

        time_since_last_activity = now - self._last_activity[user_index]

        # Show idle warning when half of the allowed idle time is elapsed.
        if time_since_last_activity >= timedelta(
            seconds=self._max_idle_duration / 2
        ):
            self._show_idle_kick_warning[user_index] = True

        # Kick when the allowed allowed idle time is elapsed.
        if time_since_last_activity >= timedelta(
            seconds=self._max_idle_duration
        ):
            print(f"User {user_index} is idle. Kicking.")
            self._remote_client_state.kick(Mask.from_index(user_index))

    def _update_frame_counter_and_display_latency(
        self, user_index: int, server_sps: float
    ) -> None:
        """Update the frame counter."""
        recent_server_keyframe_id = (
            self._remote_client_state.pop_recent_server_keyframe_id(user_index)
        )
        if recent_server_keyframe_id is not None:
            new_avg = self._client_frame_latency_avg_helper[user_index].add(
                self._frame_counter[user_index] - recent_server_keyframe_id
            )
            if new_avg and server_sps is not None:
                latency_ms = new_avg / server_sps * 1000
                self._display_latency_ms[user_index] = latency_ms
        self._client_message_manager.set_server_keyframe_id(
            self._frame_counter[user_index], Mask.from_index(user_index)
        )
        self._frame_counter[user_index] += 1

    def update(
        self, user_index: int, is_user_idle_this_frame: bool, server_sps: float
    ) -> None:
        """Update the client helper."""
        self._update_idle_kick(user_index, is_user_idle_this_frame)
        self._update_frame_counter_and_display_latency(user_index, server_sps)
