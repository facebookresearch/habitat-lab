#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from s3_upload import upload_file_to_s3
import hydra
import magnum as mn
import numpy as np
from ui import UI
from collaboration_episode_loader import load_collaboration_episode_data

from habitat.sims.habitat_simulator import sim_utilities
from habitat_hitl._internal.networking.average_rate_tracker import (
    AverageRateTracker,
)
from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.app_states.app_state_abc import AppState
from habitat_hitl.core.client_helper import ClientHelper
from habitat_hitl.core.gui_input import GuiInput
from habitat_hitl.core.hitl_main import hitl_main
from habitat_hitl.core.hydra_utils import register_hydra_plugins
from habitat_hitl.core.text_drawer import TextOnScreenAlignment
from habitat_hitl.core.types import ConnectionRecord, DisconnectionRecord
from habitat_hitl.core.user_mask import Mask, Users
from habitat_hitl.environment.camera_helper import CameraHelper
from habitat_hitl.environment.controllers.controller_abc import GuiController
from habitat_hitl.environment.controllers.gui_controller import (
    GuiHumanoidController,
    GuiRobotController,
)
from habitat_hitl.environment.hablab_utils import get_agent_art_obj_transform
from habitat_sim.utils.common import quat_from_magnum, quat_to_coeffs

class AppData():
    """RearrangeV2 data."""
    def __init__(self, max_user_count: int):
        self.max_user_count = max_user_count
        self.connected_users: Dict[int, ConnectionRecord] = {}

        self.episode_ids: Optional[List[str]] = None
        self.current_episode_index = 0

class BaseRearrangeState(AppState):
    def __init__(
            self,
            app_service: AppService,
            app_data: AppData,
        ):
            self._app_service = app_service
            self._app_data = app_data
            self._cancel = False

    def on_enter(self):
        print(f"Entering state: {type(self)}")
    def on_exit(self):
        print(f"Exiting state: {type(self)}")
    def try_cancel(self): self._cancel = True
    def get_next_state(self) -> Optional[BaseRearrangeState]: pass

    def on_environment_reset(self, episode_recorder_dict): pass
    def sim_update(self, dt: float, post_sim_update_dict): pass
    def record_state(self): pass

    def _status_message(self, message: str) -> None:
        """Send a message to all users."""
        if len(message) > 0:
            self._app_service.text_drawer.add_text(
                message,
                TextOnScreenAlignment.TOP_CENTER,
                text_delta_x=-280,
                text_delta_y=-50,
                destination_mask=Mask.ALL,
            )

class AppStateMain(AppState):
    def __init__(
            self,
            app_service: AppService,
        ):
            self._app_service = app_service
            self._app_data = AppData(app_service.hitl_config.networking.max_client_count)
            self._idle = True
            self._app_state: BaseRearrangeState = AppStateLobby(app_service, self._app_data)

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
            raise RuntimeError(f"User index {user_index} already connected! Aborting.")
        self._app_data.connected_users[connection["userIndex"]] = connection

    def _on_client_disconnected(self, disconnection: DisconnectionRecord):
        user_index = disconnection["userIndex"]
        if user_index not in self._app_data.connected_users:
            raise RuntimeError(f"User index {user_index} already disconnected! Aborting.")
        del self._app_data.connected_users[disconnection["userIndex"]]

        self._app_state.try_cancel()

    def on_environment_reset(self, episode_recorder_dict):
        self._app_state.on_environment_reset(episode_recorder_dict)

    def sim_update(self, dt: float, post_sim_update_dict):
        post_sim_update_dict["cam_transform"] = mn.Matrix4.identity_init()
        self._app_state.sim_update(dt, post_sim_update_dict)

        next_state = self._app_state.get_next_state()
        if next_state is not None:
             self._app_state.on_exit()
             self._app_state = next_state
             self._app_state.on_enter()

        # TODO: 1h maximum session time

    def record_state(self): pass  # Don't use the step recorder.


class AppStateLobby(BaseRearrangeState):
    """
    Idle state. Ends when the target user count is reached.
    """
    def __init__(self, app_service: AppService, app_data: AppData,):
        super().__init__(app_service, app_data)
    def on_enter(self): pass
    def on_exit(self): pass
    def get_next_state(self) -> Optional[BaseRearrangeState]:
        if len(self._app_data.connected_users) == self._app_data.max_user_count:
            return AppStateSession(self._app_service, self._app_data)
        return None

    def sim_update(self, dt: float, post_sim_update_dict):
        missing_users = self._app_data.max_user_count - len(self._app_data.connected_users)
        s = "s" if missing_users > 1 else ""
        message = f"Waiting for {missing_users} participant{s} to join."
        self._status_message(message)

class AppStateLoading(BaseRearrangeState):
    """
    Load next episode. Loading screen.
    Cancellable.
    """
    def __init__(self, app_service: AppService, app_data: AppData,):
        super().__init__(app_service, app_data)
        self._loading = True
        self._session_ended = False
        self._frame_number = 0
    def on_enter(self): pass
    def on_exit(self): pass
    def get_next_state(self) -> Optional[BaseRearrangeState]:
        if self._cancel:
            return AppStateCleanup(self._app_service, self._app_data)
        if self._session_ended:
            return AppStateEndSession(self._app_service, self._app_data)
        if not self._loading:
            return AppStateStartScreen(self._app_service, self._app_data)
        return None

    def sim_update(self, dt: float, post_sim_update_dict):
        self._status_message("Loading...")

        # HACK: Skip a frame so that the status message reaches the client before the server blocks.
        if self._frame_number == 1:
            if self._app_data.episode_ids is None:
                self._load_episode()
            else:
                self._increment_episode()
            self._loading = False

        self._frame_number += 1

    def _load_episode(self):
        data = self._app_data

        # Validate that all users are requesting the same episodes.
        assert len(data.connected_users) > 0
        episodes_str = data.connected_users.values()[0]["episodes"]
        for connection_record in data.connected_users.values():
            if connection_record["episodes"] != episodes_str:
                print("Users are requesting different episodes! Cancelling session.")
                self._cancel = True
                return
            
        # Validate that the episode set is not empty.
        if episodes_str is None:
            print("Users did not request episodes. Cancelling session.")
            self._cancel = True
            return
        
        # Format: {lower_bound}-{upper_bound} E.g. 100-110
        # Upper bound is exclusive.
        episode_range_str = episodes_str.split("-")
        if len(episode_range_str) != 2:
            print("Invalid episode range. Cancelling session.")
            self._cancel = True
            return
        
        # Validate that episodes are numeric.
        start_episode_id = (
            int(episode_range_str[0])
            if episode_range_str[0].isdecimal()
            else None
        )
        last_episode_id = (
            int(episode_range_str[1])
            if episode_range_str[0].isdecimal()
            else None
        )
        if start_episode_id is None or last_episode_id is None or start_episode_id < 0:
            print("Invalid episode names. Cancelling session.")
            self._cancel = True
            return
        
        total_episode_count = len(self._app_service.episode_helper._episode_iterator.episodes)

        # Validate episode range.
        if start_episode_id >= total_episode_count:
            print("Invalid episode names. Cancelling session.")
            self._cancel = True
            return
        
        last_episode_id += 1 # Hack: Links are inclusive.
        if last_episode_id >= total_episode_count:
            last_episode_id = total_episode_count

        # If in decreasing order, swap.
        if start_episode_id > last_episode_id:
            temp = last_episode_id
            last_episode_id = start_episode_id
            start_episode_id = temp
        episode_ids = []
        for episode_id_int in range(
            start_episode_id, last_episode_id
        ):
            episode_ids.append(str(episode_id_int))

        # Change episode.
        data.episode_ids = episode_ids
        data.current_episode_index = 0
        self._set_episode(data.current_episode_index)

    def _increment_episode(self):
        data = self._app_data
        assert data.episode_ids is not None
        data.current_episode_index += 1
        if data.current_episode_index < len(data.episode_ids):
            self._set_episode(data.current_episode_index)
        else:
            self._end_session(record=True)

    def _set_episode(self, episode_index: int):
        data = self._app_data
        assert data.episode_ids is not None
        if episode_index >= len(data.episode_ids):
            self._end_session()
            return
        
        next_episode_id = data.episode_ids[episode_index]
        self._app_service.episode_helper.set_next_episode_by_id(
            next_episode_id
        )
        self._app_service.end_episode(do_reset=True)

    def _end_session(self):
        self._session_ended = True


class AppStateStartScreen(BaseRearrangeState):
    """
    Start screen with a "Start" button that all users must press before starting the session.
    Cancellable.
    """
    def __init__(self, app_service: AppService, app_data: AppData,):
        super().__init__(app_service, app_data)
    def on_enter(self): pass
    def on_exit(self): pass
    def get_next_state(self) -> Optional[BaseRearrangeState]:
        if self._cancel:
            return AppStateCleanup(self._app_service, self._app_data)
        # TODO: If modal dialogue is acknowledged...
        #   return AppStateSession(self._app_service, self._app_data)
        return AppStateRearrangeV2(self._app_service, self._app_data)
        return None

    def sim_update(self, dt: float, post_sim_update_dict):
        # TODO: Top-down view.
        pass # TODO: Modal dialogue.

class AppStateSession(BaseRearrangeState):
    """
    TODO: RearrangeV2.
    Cancellable.
    """
    def __init__(self, app_service: AppService, app_data: AppData,):
        super().__init__(app_service, app_data)
        #self._app = AppStateRearrangeV2(app_service)
    def on_enter(self): pass
    def on_exit(self): pass
    def get_next_state(self) -> Optional[BaseRearrangeState]:
        if self._cancel:
            return AppStateCleanup(self._app_service, self._app_data)
        # TODO Wait for both users to terminate their task.
        # if TODO next episode exists:
        #   return AppStateStartScreen(app_service, app_data)
        # if TODO last episode:
        #   return AppStateEndSession(app_service, app_data)
        return None

    def sim_update(self, dt: float, post_sim_update_dict):
        pass

class AppStateEndSession(BaseRearrangeState):
    """
    Indicate users that the session is terminated.
    """
    def __init__(self, app_service: AppService, app_data: AppData,):
        super().__init__(app_service, app_data)
        self._elapsed_time = 0.0
        self._done = False
    def get_next_state(self) -> Optional[BaseRearrangeState]:
        if self._done:
            return AppStateUpload(self._app_service, self._app_data)
        return None
    def sim_update(self, dt: float, post_sim_update_dict):
        self._elapsed_time += dt
        if self._elapsed_time > 3.0:
            self._kick_all_users()
            self._done = True
            
        # TODO: Top-down view.
        self._status_message("Session ended successfully.")

    def _kick_all_users(self) -> None:
        "Kick the users."
        for connection_records in self._app_data.connected_users.values():
            connection_id = connection_records["connectionId"]
            self._app_service.client_message_manager.signal_kick_client(
                connection_id, Mask.ALL
            )

class AppStateUpload(BaseRearrangeState):
    """
    Upload collected data.
    """
    def __init__(self, app_service: AppService, app_data: AppData,):
        super().__init__(app_service, app_data)
        # This blocks until the upload is finished.
        # TODO: Retry if failed?

        # TODO: Query a source of truth for paths and data (call it SessionRecorder)
        #upload_file_to_s3(...)

    def get_next_state(self) -> Optional[BaseRearrangeState]:
        return AppStateCleanup(self._app_service, self._app_data)

class AppStateCleanup(BaseRearrangeState):
    """
    Kick all users and restore state for a new session.
    """
    def __init__(self, app_service: AppService, app_data: AppData,):
        super().__init__(app_service, app_data)
    def on_enter(self): pass
    def on_exit(self): pass  # TODO: Clean-up app_data.
    def get_next_state(self) -> Optional[BaseRearrangeState]:
        # TODO: Wait for output folder to be cleaned.
        # TODO: Wait for data to be uploaded to S3.

        # Wait for users to be kicked.
        if len(self._app_data.connected_users) == 0:
            # Create a fresh AppData
            new_app_data = AppData(self._app_data.max_user_count)
            return AppStateLobby(self._app_service, new_app_data)
        else: return None

    def sim_update(self, dt: float, post_sim_update_dict):
        # TODO: Get error message from app_data, and wait a frame until kicking

        self._kick_all_users()
    


    def _kick_all_users(self) -> None:
        # TODO: Move to util
        "Kick the users."
        for connection_records in self._app_data.connected_users.values():
            user_index = connection_records["userIndex"]
            self._app_service.remote_client_state.kick(user_index)


@hydra.main(
    version_base=None, config_path="config", config_name="rearrange_v2"
)
def main(config):
    hitl_main(
        config,
        lambda app_service: AppStateMain(app_service),
    )

if __name__ == "__main__":
    register_hydra_plugins()
    main()