#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from habitat_hitl.core.average_helper import AverageHelper


class ClientHelper:
    """
    Tracks connected remote clients. Displays client latency and kicks idle clients.
    """

    def __init__(self, app_service):
        self._app_service = app_service
        self._show_idle_kick_warning = False
        self._idle_frame_counter = None
        self._frame_counter = 0

        self._client_frame_latency_avg_helper = AverageHelper(
            window_size=10, output_rate=10
        )
        self._display_latency_ms = None

    @property
    def display_latency_ms(self):
        return self._display_latency_ms

    @property
    def do_show_idle_kick_warning(self):
        return self._show_idle_kick_warning

    def _update_idle_kick(self, is_user_idle_this_frame):
        hitl_config = self._app_service.hitl_config
        self._show_idle_kick_warning = False

        connection_records = (
            self._app_service.remote_client_state.get_new_connection_records()
        )
        if len(connection_records):
            assert (
                len(connection_records) == 1
            )  # todo: expand this to support multiple connections
            connection_record = connection_records[-1]
            # new connection
            self._client_connection_id = connection_record["connectionId"]
            print(f"new connection_record: {connection_record}")
            if hitl_config.networking.client_max_idle_duration is not None:
                self._idle_frame_counter = 0

        if self._idle_frame_counter is not None:
            if is_user_idle_this_frame:
                self._idle_frame_counter += 1
                assert hitl_config.networking.client_max_idle_duration > 0
                max_idle_frames = max(
                    int(
                        hitl_config.networking.client_max_idle_duration
                        * hitl_config.target_sps
                    ),
                    1,
                )

                if self._idle_frame_counter > max_idle_frames * 0.5:
                    self._show_idle_kick_warning = True

                if self._idle_frame_counter > max_idle_frames:
                    self._app_service.client_message_manager.signal_kick_client(
                        self._client_connection_id
                    )
                    self._idle_frame_counter = None
            else:
                # reset counter whenever the client isn't idle
                self._idle_frame_counter = 0

    def _update_frame_counter_and_display_latency(self, server_sps):
        recent_server_keyframe_id = (
            self._app_service.remote_client_state.pop_recent_server_keyframe_id()
        )
        if recent_server_keyframe_id is not None:
            new_avg = self._client_frame_latency_avg_helper.add(
                self._frame_counter - recent_server_keyframe_id
            )
            if new_avg and server_sps is not None:
                latency_ms = new_avg / server_sps * 1000
                self._display_latency_ms = latency_ms

        if self._app_service.client_message_manager:
            self._app_service.client_message_manager.set_server_keyframe_id(
                self._frame_counter
            )
        self._frame_counter += 1

    def update(self, is_user_idle_this_frame, server_sps):
        self._update_idle_kick(is_user_idle_this_frame)
        self._update_frame_counter_and_display_latency(server_sps)
