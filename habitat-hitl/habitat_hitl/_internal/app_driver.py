#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import json
from typing import Any, Optional

from habitat_hitl._internal.networking.interprocess_record import (
    InterprocessRecord,
)
from habitat_hitl._internal.networking.keyframe_utils import get_empty_keyframe
from habitat_hitl._internal.networking.networking_process import (
    launch_networking_process,
    terminate_networking_process,
)
from habitat_hitl.core.client_message_manager import ClientMessageManager
from habitat_hitl.core.gui_drawer import GuiDrawer
from habitat_hitl.core.gui_input import GuiInput
from habitat_hitl.core.hydra_utils import omegaconf_to_object
from habitat_hitl.core.remote_client_state import RemoteClientState
from habitat_hitl.core.text_drawer import AbstractTextDrawer
from habitat_hitl.core.types import KeyframeAndMessages
from habitat_hitl.core.ui_elements import UIManager
from habitat_hitl.core.user_mask import Users
from habitat_sim.gfx import DebugLineRender


class AppDriver:
    @abc.abstractmethod
    def sim_update(self, dt: float):
        pass

    def __init__(
        self,
        config,
        gui_input: GuiInput,
        line_render: Optional[DebugLineRender],
        text_drawer: AbstractTextDrawer,
        sim: Any,
    ):
        """
        Base HITL application driver.
        """
        if "habitat_hitl" not in config:
            raise RuntimeError(
                "Required parameter 'habitat_hitl' not found in config. See hitl_defaults.yaml."
            )
        self._hitl_config = omegaconf_to_object(config.habitat_hitl)
        self._gui_input = gui_input
        self._sim = sim

        # Create a user container.
        # In local mode, there is always 1 active user.
        # In remote mode, use 'activate_user()' and 'deactivate_user()' when handling connections.
        self._users = Users(
            max_user_count=max(
                self._hitl_config.networking.max_client_count, 1
            ),
            activate_users=not self._hitl_config.networking.enable,
        )

        # Initialize client message manager.
        self._client_message_manager: Optional[ClientMessageManager] = None
        if self.network_server_enabled:
            self._client_message_manager = ClientMessageManager(self._users)

        # Initialize GUI Drawer.
        self._gui_drawer = GuiDrawer(line_render, self._client_message_manager)
        self._gui_drawer.set_line_width(self._hitl_config.debug_line_width)

        # Initialize networking
        self._remote_client_state: Optional[RemoteClientState] = None
        self._interprocess_record: Optional[InterprocessRecord] = None
        if self.network_server_enabled:
            self._interprocess_record = InterprocessRecord(
                self._hitl_config.networking
            )
            launch_networking_process(self._interprocess_record)
            self._remote_client_state = RemoteClientState(
                hitl_config=self._hitl_config,
                client_message_manager=self._client_message_manager,
                interprocess_record=self._interprocess_record,
                gui_drawer=self._gui_drawer,
                users=self._users,
            )
            # Bind the server input to user 0
            if self._hitl_config.networking.client_sync.server_input:
                self._remote_client_state.bind_gui_input(self._gui_input, 0)

        # Limit the number of float decimals in JSON transmissions
        self.get_sim().gfx_replay_manager.set_max_decimal_places(4)

        # Initialize UI utilities.
        self._text_drawer = text_drawer
        # TODO: Dependency injection
        text_drawer._client_message_manager = self._client_message_manager
        self._ui_manager = UIManager(
            self._users,
            self._remote_client_state,
            self._client_message_manager,
        )

    def close(self):
        if self.network_server_enabled:
            terminate_networking_process()

    def get_sim(self) -> Any:
        return self._sim

    def _send_keyframes(self, keyframes_json: list[str]):
        assert self.network_server_enabled

        keyframes = []
        for keyframe_json in keyframes_json:
            obj = json.loads(keyframe_json)
            assert "keyframe" in obj
            keyframe_obj = obj["keyframe"]
            keyframes.append(keyframe_obj)

        # If messages need to be sent, but no keyframe is available, create an empty keyframe.
        if self._client_message_manager.any_message() and len(keyframes) == 0:
            keyframes.append(get_empty_keyframe())

        for keyframe in keyframes:
            # Remove rigs from keyframe if skinning is disabled.
            if not self._hitl_config.networking.client_sync.skinning:
                if "rigCreations" in keyframe:
                    del keyframe["rigCreations"]
                if "rigUpdates" in keyframe:
                    del keyframe["rigUpdates"]
            # Insert server->client message into the keyframe.
            messages = self._client_message_manager.get_messages()
            self._client_message_manager.clear_messages()
            # Send the keyframe.
            self._interprocess_record.send_keyframe_to_networking_thread(
                KeyframeAndMessages(keyframe, messages)
            )

    @property
    def network_server_enabled(self) -> bool:
        return (
            self._hitl_config.networking.enable
            and self._hitl_config.networking.max_client_count > 0
        )
