#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations
import os
import shutil
from time import time
from typing import Any, Dict, List, Optional, Tuple

from habitat_hitl.core.serialize_utils import save_as_json_gzip
from habitat_hitl.environment.hitl_tutorial import _lookat_bounding_box_top_down

# Importing this file enables the collaboration dataset episode type to be loaded.
from collaboration_episode_loader import load_collaboration_episode_data

from habitat_hitl.core.client_message_manager import UIButton
from habitat_hitl.core.key_mapping import KeyCode
import hydra
import magnum as mn
import numpy as np
from s3_upload import upload_file_to_s3
from ui import UI, World

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

UP = mn.Vector3(0, 1, 0)
FWD = mn.Vector3(0, 0, 1)

def timestamp() -> str:
    "Generate a Unix timestamp at the current time."
    return str(int(time()))

def get_top_down_view(sim) -> mn.Matrix4:
    scene_root_node = sim.get_active_scene_graph().get_root_node()
    scene_target_bb: mn.Range3D = scene_root_node.cumulative_bb
    look_at = _lookat_bounding_box_top_down(
        100, scene_target_bb, FWD
    )
    return mn.Matrix4.look_at(look_at[0], look_at[1], UP)

class Session():
    """
    RearrangeV2 session.
    """
    def __init__(
            self,
            config: Dict[str, Any],
            episode_ids: List[str],
            connection_records: Dict[int, ConnectionRecord],
        ):
        self.success = False
        self.episode_ids = episode_ids
        self.current_episode_index = 0
        self.connection_records = connection_records
        self.session_recorder = SessionRecorder(config, connection_records, episode_ids)
        self.status = ""  # Use this to display error status

        # Use the port as a discriminator for when there are multiple concurrent servers.
        output_folder_suffix = str(config.habitat_hitl.networking.port)
        self.output_folder = f"output_{output_folder_suffix}"

class SessionRecorder():
    def __init__(
        self,
        config: Dict[str, Any],
        connection_records: Dict[int, ConnectionRecord],
        episode_ids: List[str],
    ):
        self.data = {
            "episode_ids": episode_ids,
            "completed": False,
            "error": "",
            "start_timestamp": timestamp(),
            "end_timestamp": timestamp(),
            "config": config,
            "frame_count": 0,
            "users": [],
            "episodes": [],
        }
        
        for user_index in range(len(connection_records)):
            self.data["users"].append({
                "user_index": user_index,
                #"agent_index": TODO: Only available during rearrange.
                "connection_record": connection_records[user_index],
            })

    def end_session(self, error: str):
        self.data["end_timestamp"] = timestamp()
        self.data["completed"] = True
        self.data["error"] = error

    def start_episode(
        self,
        episode_id: str,
        scene_id: str,
        dataset: str,
    ):
        self.data["episodes"].append({
            "episode_id": episode_id,
            "scene_id": scene_id,
            "start_timestamp": timestamp(),
            "end_timestamp": timestamp(),
            "completed": False,
            "success": False,
            "frame_count": 0,
            "dataset": dataset,
            "frames": [],
        })

    def end_episode(
        self,
        success: bool,
    ):
        self.data["episodes"][-1]["end_timestamp"] = timestamp()
        self.data["episodes"][-1]["success"] = success
        self.data["episodes"][-1]["completed"] = True

    def record_frame(
        self,
        frame_data: Dict[str, Any],
    ):
        self.data["end_timestamp"] = timestamp()
        self.data["frame_count"] += 1

        self.data["episodes"][-1]["end_timestamp"] = timestamp()
        self.data["episodes"][-1]["frame_count"] += 1
        self.data["episodes"][-1]["frames"].append(frame_data)
        
class AppData():
    """RearrangeV2 data."""
    def __init__(self, max_user_count: int):
        self.max_user_count = max_user_count
        self.connected_users: Dict[int, ConnectionRecord] = {}
        self.session: Optional[Session] = None

        self.episode_ids: Optional[List[str]] = None
        self.current_episode_index = 0

    def reset(self):
        assert len(self.connected_users) == 0, "Cannot reset RearrangeV2 state if users are still connected!"
        self.episode_ids = None
        self.current_episode_index = 0
        self.session = None
        # Note: Clearing connected users is done by kicking.

class FrameRecorder:
    def __init__(self, app_service: AppService, app_data: AppData, world: World):
        self._app_service = app_service
        self._app_data = app_data
        self._sim = app_service.sim
        self._world = world

    def get_num_agents(self):
        return len(self._sim.agents_mgr._all_agent_data)

    def get_agents_state(self):
        agent_states = []
        for agent_idx in range(self.get_num_agents()):
            agent_root = get_agent_art_obj_transform(self._sim, agent_idx)
            position = np.array(agent_root.translation).tolist()
            rotation = mn.Quaternion.from_matrix(agent_root.rotation())
            rotation = quat_to_coeffs(quat_from_magnum(rotation)).tolist()

            snap_idx = self._sim.agents_mgr._all_agent_data[
                agent_idx
            ].grasp_mgr.snap_idx
            agent_states.append(
                {
                    "position": position,
                    "rotation": rotation,
                    "grasp_mgr_snap_idx": snap_idx,
                }
            )
        return agent_states

    def get_objects_state(self):
        object_states = []
        rom = self._sim.get_rigid_object_manager()
        for object_handle, rel_idx in self._sim._handle_to_object_id.items():
            obj_id = self._sim._scene_obj_ids[rel_idx]
            ro = rom.get_object_by_id(obj_id)
            position = np.array(ro.translation).tolist()
            rotation = quat_to_coeffs(quat_from_magnum(ro.rotation)).tolist()
            object_states.append(
                {
                    "position": position,
                    "rotation": rotation,
                    "object_handle": object_handle,
                    "object_id": obj_id,
                }
            )
        return object_states
    
    def get_users_state(self) -> List[Dict[str, Any]]:
        output = []
        for user_index in range(self._app_data.max_user_count):
            #output.append({
            #    self._app_service.
            #})
            # Get camera transforms
            pass
        return output

    def record_state(self, elapsed_time:float, user_data: List[UserData]) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "t": elapsed_time,
            "users": [],
            "object_states": self.get_objects_state(),
            "agent_states": self.get_agents_state(),
        }

        for user_index in range(len(user_data)):
            u = user_data[user_index]
            user_data_dict = {
                "task_completed": u.task_completed,
                "camera_transform": u.cam_transform,
                "held_object": u.ui._held_object_id,
                #"agent_state: ...
                "events": list(u.ui._events)
            }
            data["users"].append(user_data_dict)
            u.ui._events.clear() # Sloppy

        return data

class BaseRearrangeState(AppState):
    
    def __init__(
            self,
            app_service: AppService,
            app_data: AppData,
        ):
            self._app_service = app_service
            self._app_data = app_data
            self._cancel = False
            self._time_since_last_connection = 0

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

    def _kick_all_users(self) -> None:
        "Kick all users."
        self._app_service.remote_client_state.kick(Mask.ALL)

class AppStateMain(AppState):
    """
    Main RearrangeV2 application state.
    """
    def __init__(
            self,
            app_service: AppService,
        ):
            self._app_service = app_service
            self._app_data = AppData(app_service.hitl_config.networking.max_client_count)
            self._app_state: BaseRearrangeState = AppStateReset(app_service, self._app_data)

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
        self._app_state._time_since_last_connection = 0.0

    def _on_client_disconnected(self, disconnection: DisconnectionRecord):
        user_index = disconnection["userIndex"]
        if user_index not in self._app_data.connected_users:
            # TODO: Investigate why clients keep connecting/disconnecting.
            print(f"User index {user_index} already disconnected!")
            #raise RuntimeError(f"User index {user_index} already disconnected! Aborting.")
        else:
            del self._app_data.connected_users[user_index]

        # If a user has disconnected, send a cancellation signal to the current state.
        self._app_state.try_cancel()

    def on_environment_reset(self, episode_recorder_dict):
        self._app_state.on_environment_reset(episode_recorder_dict)

    def sim_update(self, dt: float, post_sim_update_dict):
        self._app_state._time_since_last_connection += dt
        post_sim_update_dict["cam_transform"] = mn.Matrix4.identity_init()
        self._app_state.sim_update(dt, post_sim_update_dict)

        next_state = self._app_state.get_next_state()
        if next_state is not None:
             self._app_state.on_exit()
             self._app_state = next_state
             self._app_state.on_enter()

    def record_state(self): pass  # Unused.


class AppStateLobby(BaseRearrangeState):
    """
    Idle state.
    Ends when the target user count is reached.
    """
    def __init__(self, app_service: AppService, app_data: AppData,):
        super().__init__(app_service, app_data)

    def on_enter(self):
        super().on_enter()
        # Enable new connections
        # Sloppy: Create API
        self._app_service._remote_client_state._interprocess_record.enable_new_connections(True)
    
    def on_exit(self):
        super().on_exit()
        # Disable new connections
        # Sloppy: Create API
        self._app_service._remote_client_state._interprocess_record.enable_new_connections(False)


    def get_next_state(self) -> Optional[BaseRearrangeState]:
        if len(self._app_data.connected_users) == self._app_data.max_user_count and self._time_since_last_connection > 0.5:
            return AppStateStartSession(self._app_service, self._app_data)
        return None

    def sim_update(self, dt: float, post_sim_update_dict):
        missing_users = self._app_data.max_user_count - len(self._app_data.connected_users)
        s = "s" if missing_users > 1 else ""
        message = f"Waiting for {missing_users} participant{s} to join."
        self._status_message(message)


class AppStateStartSession(BaseRearrangeState):
    def __init__(self, app_service: AppService, app_data: AppData,):
        super().__init__(app_service, app_data)

    def on_enter(self):
        super().on_enter()

        if self._try_get_episodes():
            # Start the session.
            self._app_data.session = Session(
                self._app_service.config,
                self._app_data.episode_ids,
                self._app_data.connected_users,
            )
        else:
            # Create partial session record.
            self._app_data.session = Session(
                self._app_service.config,
                [],
                self._app_data.connected_users,
            )
            self._cancel = True

    def get_next_state(self) -> Optional[BaseRearrangeState]:
        if self._cancel:
            return AppStateEndSession(self._app_service, self._app_data, "Invalid session")
        return AppStateLoadEpisode(self._app_service, self._app_data)

    def _try_get_episodes(self):
        data = self._app_data

        # Validate that episodes are selected.
        assert len(data.connected_users) > 0
        connection_record = episodes_str = list(data.connected_users.values())[0]
        if "episodes" not in connection_record:
            print("Users did not request episodes. Cancelling session.")
            return False
        episodes_str = connection_record["episodes"]

        # Validate that all users are requesting the same episodes.
        for connection_record in data.connected_users.values():
            if connection_record["episodes"] != episodes_str:
                print("Users are requesting different episodes! Cancelling session.")
                return False
            
        # Validate that the episode set is not empty.
        if episodes_str is None or len(episodes_str) == 0:
            print("Users did not request episodes. Cancelling session.")
            return False
        
        # Format: {lower_bound}-{upper_bound} E.g. 100-110
        # Upper bound is exclusive.
        episode_range_str = episodes_str.split("-")
        if len(episode_range_str) != 2:
            print("Invalid episode range. Cancelling session.")
            return False
        
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
            return False
        
        total_episode_count = len(self._app_service.episode_helper._episode_iterator.episodes)

        # Validate episode range.
        if start_episode_id >= total_episode_count:
            print("Invalid episode names. Cancelling session.")
            return False
        
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
        return True

class AppStateLoadEpisode(BaseRearrangeState):
    """
    Load an episode.
    A loading screen is displaying while the content loads.
    * If no episode set is selected, create a new episode set from connection record.
    * If a next episode exists, fires up RearrangeV2.
    * If all episodes are done, end session.
    Cancellable.
    """
    def __init__(self, app_service: AppService, app_data: AppData):
        super().__init__(app_service, app_data)
        self._loading = True
        self._session_ended = False
        self._frame_number = 0

    def get_next_state(self) -> Optional[BaseRearrangeState]:
        if self._cancel:
            # TODO: Error management
            return AppStateEndSession(self._app_service, self._app_data, "User disconnected")
        if self._session_ended:
            return AppStateEndSession(self._app_service, self._app_data)
        if not self._loading:
            return AppStateStartScreen(self._app_service, self._app_data)
        return None

    def sim_update(self, dt: float, post_sim_update_dict):
        self._status_message("Loading...")

        # HACK: Skip a frame so that the status message reaches the client before the server blocks.
        # TODO: Clean this up.
        if self._frame_number == 1:
            self._increment_episode()
            self._loading = False

        self._frame_number += 1

    def _increment_episode(self):
        data = self._app_data
        assert data.episode_ids is not None
        if data.current_episode_index < len(data.episode_ids):
            self._set_episode(data.current_episode_index)
            data.current_episode_index += 1
        else:
            self._session_ended = True

    def _set_episode(self, episode_index: int):
        data = self._app_data

        # Set the ID of the next episode to play in lab.
        next_episode_id = data.episode_ids[episode_index]
        self._app_service.episode_helper.set_next_episode_by_id(
            next_episode_id
        )
        
        # Once an episode ID has been set, lab needs to be reset to load the episode.
        self._app_service.end_episode(do_reset=True)

        # Insert a keyframe to force clients to load immediately.
        self._app_service.sim.gfx_replay_manager.save_keyframe()

        # TODO: Wait for clients to finish loading before starting rearrange.
        # TODO: Timeout

START_BUTTON_ID = "start"
START_SCREEN_TIMEOUT = 120.0
SKIP_START_SCREEN = False
class AppStateStartScreen(BaseRearrangeState):
    """
    Start screen with a "Start" button that all users must press before starting the session.
    Cancellable.
    """
    def __init__(self, app_service: AppService, app_data: AppData,):
        super().__init__(app_service, app_data)
        self._ready_to_start: List[bool] = [False] * self._app_data.max_user_count
        self._elapsed_time: float = 0.0
        self._timeout = False  # TODO: Error management

    def get_next_state(self) -> Optional[BaseRearrangeState]:
        if self._cancel:
            return AppStateEndSession(self._app_service, self._app_data, "Timeout" if self._timeout else "User disconnected")
        
        # If all users pressed the "Start" button, begin the session.
        ready_to_start = True
        for user_ready in self._ready_to_start:
            ready_to_start &= user_ready
        if ready_to_start or SKIP_START_SCREEN:
            return AppStateRearrangeV2(self._app_service, self._app_data)
        
        return None

    def sim_update(self, dt: float, post_sim_update_dict):
        # Top-down view.
        cam_matrix = get_top_down_view(self._app_service.sim)
        post_sim_update_dict["cam_transform"] = cam_matrix
        self._app_service._client_message_manager.update_camera_transform(
            cam_matrix, destination_mask=Mask.ALL
        )

        # Time limit to start the experiment.
        self._elapsed_time += dt
        remaining_time = START_SCREEN_TIMEOUT - self._elapsed_time
        if remaining_time <= 0:
            self._cancel = True
            self._timeout = True
            return
        remaining_time_int = int(remaining_time)
        title = f"New Session (Expires in: {remaining_time_int}s)"

        # Show dialogue box with "Start" button.
        for user_index in range(self._app_data.max_user_count):
            button_pressed = self._app_service.remote_client_state.ui_button_clicked(user_index, START_BUTTON_ID)
            self._ready_to_start[user_index] |= button_pressed

            if not self._ready_to_start[user_index]:
                self._app_service.client_message_manager.show_modal_dialogue_box(
                    title,
                    "Press 'Start' to begin the experiment.",
                    [UIButton(START_BUTTON_ID, "Start", True)],
                    Mask.from_index(user_index),
                )
            else:
                self._app_service.client_message_manager.show_modal_dialogue_box(
                    title,
                    "Waiting for other participants...",
                    [UIButton(START_BUTTON_ID, "Start", False)],
                    Mask.from_index(user_index),
                )

        # Server-only: Press numeric keys to press button on behalf of users.
        if not self._app_service.hitl_config.experimental.headless.do_headless:
            server_message = "Press numeric keys to start on behalf of users."
            first_key = int(KeyCode.ONE)
            for user_index in range(len(self._ready_to_start)):
                if self._app_service.gui_input.get_key_down(KeyCode(first_key + user_index)):
                    self._ready_to_start[user_index] = True
                user_ready = self._ready_to_start[user_index]
                server_message += f"\nUser {user_index}: {'Ready' if user_ready else 'Not ready'}."

            self._app_service.text_drawer.add_text(
                server_message,
                TextOnScreenAlignment.TOP_LEFT,
                text_delta_x=0,
                text_delta_y=-50,
                destination_mask=Mask.NONE,
            )


SESSION_END_DELAY = 5.0
class AppStateEndSession(BaseRearrangeState):
    """
    Indicate users that the session is terminated.
    """
    def __init__(self, app_service: AppService, app_data: AppData, error_message: str = ""):
        super().__init__(app_service, app_data)
        self._elapsed_time = 0.0
        self._next_state: Optional[BaseRearrangeState] = None

        # TODO: Gather from session.
        self._error = error_message
        self._status = "Session ended."
        if len(error_message) > 0:
            self._status += f"Error: {error_message}"

    def get_next_state(self) -> Optional[BaseRearrangeState]:
        return self._next_state
    
    def sim_update(self, dt: float, post_sim_update_dict):
        # Top-down view.
        cam_matrix = get_top_down_view(self._app_service.sim)
        post_sim_update_dict["cam_transform"] = cam_matrix
        self._app_service._client_message_manager.update_camera_transform(
            cam_matrix, destination_mask=Mask.ALL
        )

        self._status_message(self._status)
        self._elapsed_time += dt
        if self._elapsed_time > SESSION_END_DELAY:
            self._end_session()
            self._next_state = AppStateReset(self._app_service, self._app_data)

    def _end_session(self):
        session = self._app_data.session
        if session is None:
            print("Null session. Skipping S3 upload.")
            return

        # Finalize session.
        if self._error == "":
            session.success = True
        session.session_recorder.end_session(self._error)

        # Find S3 params.
        data_collection_config = self._app_service.config.rearrange_v2.data_collection
        s3_path = data_collection_config.s3_path
        if s3_path[-1] != "/":
            s3_path += "/"
        s3_subdir = "complete" if session.success else "incomplete"
        s3_path += s3_subdir

        # Delete previous output directory
        output_folder = session.output_folder
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        
        # Create new output directory
        os.makedirs(output_folder)
        json_path = os.path.join(output_folder, "session.json.gz")
        save_as_json_gzip(session.session_recorder, json_path)

        # Generate unique session ID
        if len(self._app_data.session.connection_records) == 0:
            print("No connection record. Aborting upload.")
            return        
        episodes_str = ""
        user_id_str = ""
        for _, connection_record in self._app_data.session.connection_records.items():
            if "episodes" in connection_record:
                episodes_str = connection_record["episodes"]
            else:
                episodes_str = "unknown_episodes"
            
            if "user_id" in connection_record:
                if user_id_str != "":
                    user_id_str += "_"
                user_id_str += connection_record["user_id"]
            else:
                user_id_str += "unknown_user"
        session_id = f"{episodes_str}_{user_id_str}_{timestamp()}"
        # TODO Filter str

        # Upload output directory
        output_files = [f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f))]
        for output_file in output_files:
            local_file_path = os.path.join(output_folder, output_file)
            s3_file_name = f"{session_id}_{output_file}"
            upload_file_to_s3(local_file_path, s3_file_name, s3_path)

class AppStateReset(BaseRearrangeState):
    """
    Kick all users and restore state for a new session.
    Actions:
    * Kick all users.
    * Clean-up AppData.
    """
    def __init__(self, app_service: AppService, app_data: AppData,):
        super().__init__(app_service, app_data)

    def on_enter(self):
        super().on_enter()

        # Kick all users.
        self._kick_all_users()
    
    def get_next_state(self) -> Optional[BaseRearrangeState]:
        # Wait for users to be kicked.
        if (len(self._app_data.connected_users) == 0):
            # Clean-up AppData before opening the lobby.
            self._app_data.reset()

            return AppStateLobby(self._app_service, self._app_data)
        else: return None

class UserData:
    def __init__(
        self,
        app_service: AppService,
        user_index: int,
        world: World,
        gui_agent_controller: GuiController,
        server_sps_tracker: AverageRateTracker,
    ):
        self.app_service = app_service
        self.user_index = user_index
        self.gui_agent_controller = gui_agent_controller
        self.server_sps_tracker = server_sps_tracker
        self.gui_input: GuiInput = app_service.remote_client_state.get_gui_input(
            user_index
        )
        self.cam_transform = mn.Matrix4.identity_init()
        self.show_gui_text = True
        self.task_instruction = ""
        self.task_completed = False

        self.camera_helper = CameraHelper(
            app_service.hitl_config,
            self.gui_input,
        )

        self.ui = UI(
            hitl_config=app_service.hitl_config,
            user_index=user_index,
            world=world,
            gui_controller=gui_agent_controller,
            sim=app_service.sim,
            gui_input=self.gui_input,
            gui_drawer=app_service.gui_drawer,
            camera_helper=self.camera_helper,
        )

        # HACK: Work around GuiController input.
        # TODO: Communicate to the controller via action hints.
        gui_agent_controller._gui_input = self.gui_input

    def reset(
        self, object_receptacle_pairs: List[Tuple[List[int], List[int]]]
    ):
        self.task_completed = False
        self.camera_helper.update(self._get_camera_lookat_pos(), dt=0)
        self.ui.reset(object_receptacle_pairs)

    def update(self, dt: float):
        if self.gui_input.get_key_down(GuiInput.KeyNS.H):
            self.show_gui_text = not self.show_gui_text

        if self.gui_input.get_key_down(GuiInput.KeyNS.ZERO):
            self.task_completed = True

        self.app_service.remote_client_state._client_helper.update(
            self.user_index,
            self._is_user_idle_this_frame(),
            self.server_sps_tracker.get_smoothed_rate(),
        )

        self.camera_helper.update(self._get_camera_lookat_pos(), dt)
        self.cam_transform = self.camera_helper.get_cam_transform()

        if self.app_service.hitl_config.networking.enable:
            self.app_service._client_message_manager.update_camera_transform(
                self.cam_transform, destination_mask=Mask.from_index(self.user_index)
            )

        self.ui.update()
        self.ui.draw_ui()

    def _get_camera_lookat_pos(self) -> mn.Vector3:
        agent_root = get_agent_art_obj_transform(
            self.app_service.sim,
            self.gui_agent_controller._agent_idx,
        )
        lookat_y_offset = UP
        lookat = agent_root.translation + lookat_y_offset
        return lookat

    def _is_user_idle_this_frame(self) -> bool:
        return not self.gui_input.get_any_input()


class AppStateRearrangeV2(BaseRearrangeState):
    """
    Todo
    """

    def __init__(self, app_service: AppService, app_data: AppData):
        super().__init__(app_service, app_data)
        # We don't sync the server camera. Instead, we maintain one camera per user.
        assert (
            app_service.hitl_config.networking.client_sync.server_camera
            == False
        )

        self._app_service = app_service
        self._gui_agent_controllers = self._app_service.gui_agent_controllers
        self._num_agents = len(self._gui_agent_controllers)
        # TODO: Move to service
        self._users = Users(
            self._app_service.hitl_config.networking.max_client_count
        )

        self._sps_tracker = AverageRateTracker(2.0)
        self._server_user_index = 0
        self._server_gui_input = self._app_service.gui_input
        self._server_input_enabled = False
        self._elapsed_time = 0.0

        self._user_data: List[UserData] = []

        self._world = World(app_service.sim)

        self._frame_recorder =  FrameRecorder(app_service, app_data, self._world)

        for user_index in self._users.indices(Mask.ALL):
            self._user_data.append(
                UserData(
                    app_service=app_service,
                    user_index=user_index,
                    world=self._world,
                    gui_agent_controller=self._gui_agent_controllers[user_index],
                    server_sps_tracker=self._sps_tracker,
                )
            )

        # Reset the environment immediately.
        self.on_environment_reset(None)

    def get_next_state(self) -> Optional[BaseRearrangeState]:
        # If cancelled, skip upload and clean-up.
        if self._cancel:
            return AppStateEndSession(self._app_service, self._app_data)

        # Check if all users signaled to terminate episode.
        change_episode = self._is_episode_finished()

        # If changing episode, go back to the loading screen.
        # This state takes care of selecting the next episode.
        if change_episode:
            return AppStateLoadEpisode(self._app_service, self._app_data)
        else:
            return None

    def on_enter(self):
        super().on_enter()
        episode = self._app_service.episode_helper.current_episode
        self._app_data.session.session_recorder.start_episode(
            episode.episode_id,
            episode.scene_id,
            episode.scene_dataset_config,
        )

    def on_exit(self):
        super().on_exit()

        self._app_data.session.session_recorder.end_episode(
            success=self._is_episode_finished()
        )

    def _is_episode_finished(self) -> bool:
        """
        Determines whether all users have finished their tasks.
        """
        for user_index in self._users.indices(Mask.ALL):
            if not self._user_data[user_index].task_completed:
                return False
        return True

    def on_environment_reset(self, episode_recorder_dict):
        self._world.reset()

        # Set the task instruction
        current_episode = self._app_service.env.current_episode
        if current_episode.info.get("instruction") is not None:
            task_instruction = current_episode.info["instruction"]
            # TODO: Users will have different instructions.
            for user_index in self._users.indices(Mask.ALL):
                self._user_data[user_index].task_instruction = task_instruction

        client_message_manager = self._app_service.client_message_manager
        if client_message_manager:
            client_message_manager.signal_scene_change(Mask.ALL)

        object_receptacle_pairs = self._create_goal_object_receptacle_pairs()
        for user_index in self._users.indices(Mask.ALL):
            self._user_data[user_index].reset(object_receptacle_pairs)

        # Insert a keyframe to force clients to load immediately.
        self._app_service.sim.gfx_replay_manager.save_keyframe()

    def _create_goal_object_receptacle_pairs(
        self,
    ) -> List[Tuple[List[int], List[int]]]:
        """Parse the current episode and returns the goal object-receptacle pairs."""
        sim = self._app_service.sim
        paired_goal_ids: List[Tuple[List[int], List[int]]] = []
        current_episode = self._app_service.env.current_episode
        if current_episode.info.get("evaluation_propositions") is not None:
            evaluation_propositions = current_episode.info["evaluation_propositions"]
            for proposition in evaluation_propositions:
                object_ids: List[int] = []
                object_handles = proposition["args"]["object_handles"]
                for object_handle in object_handles:
                    obj = sim_utilities.get_obj_from_handle(sim, object_handle)
                    object_id = obj.object_id
                    object_ids.append(object_id)
                receptacle_ids: List[int] = []
                receptacle_handles = proposition["args"]["receptacle_handles"]
                for receptacle_handle in receptacle_handles:
                    obj = sim_utilities.get_obj_from_handle(
                        sim, receptacle_handle
                    )
                    object_id = obj.object_id
                    # TODO: Support for finding links by handle.
                    receptacle_ids.append(object_id)
                paired_goal_ids.append((object_ids, receptacle_ids))
        return paired_goal_ids

    def _update_grasping_and_set_act_hints(self, user_index: int):
        gui_agent_controller = self._user_data[user_index].gui_agent_controller
        assert isinstance(
            gui_agent_controller, (GuiHumanoidController, GuiRobotController)
        )
        gui_agent_controller.set_act_hints(
            walk_dir=None,
            distance_multiplier=1.0,
            grasp_obj_idx=None,
            do_drop=None,
            cam_yaw=self._user_data[
                user_index
            ].camera_helper.lookat_offset_yaw,
            throw_vel=None,
            reach_pos=None,
        )

    def _get_gui_controlled_agent_index(self, user_index):
        return self._gui_agent_controllers[user_index]._agent_idx

    def _get_controls_text(self, user_index: int):
        if not self._user_data[user_index].show_gui_text:
            return ""

        controls_str: str = ""
        controls_str += "H: Toggle help\n"
        controls_str += "Look: Middle click (drag), I, K\n"
        controls_str += "Walk: W, S\n"
        controls_str += "Turn: A, D\n"
        controls_str += "Finish episode: Zero (0)\n"
        controls_str += "Open/close: Double-click\n"
        controls_str += "Pick object: Double-click\n"
        controls_str += "Place object: Right click (hold)\n"
        return controls_str

    def _get_status_text(self, user_index: int):
        status_str = ""

        if len(self._user_data[user_index].task_instruction) > 0:
            status_str += (
                "Instruction: "
                + self._user_data[user_index].task_instruction
                + "\n"
            )
        if self._app_service.remote_client_state._client_helper.do_show_idle_kick_warning(user_index):
            status_str += (
                "\n\nAre you still there?\nPress any key to keep playing!\n"
            )

        return status_str

    def _update_help_text(self, user_index: int):
        # If the user has signaled to change episode, show dialogue.
        if self._user_data[user_index].task_completed:
            ui_button_id = "undo_change_episode"
            self._app_service.client_message_manager.show_modal_dialogue_box(
                "Task Finished",
                "Waiting for the other participant to finish...",
                [UIButton(ui_button_id, "Cancel", True)],
                Mask.from_index(user_index),
            )
            cancel = self._app_service.remote_client_state.ui_button_clicked(user_index, ui_button_id)
            if cancel:
                self._user_data[user_index].task_completed = False
            return

        status_str = self._get_status_text(user_index)
        if len(status_str) > 0:
            self._app_service.text_drawer.add_text(
                status_str,
                TextOnScreenAlignment.TOP_CENTER,
                text_delta_x=-280,
                text_delta_y=-50,
                destination_mask=Mask.from_index(user_index),
            )

        controls_str = self._get_controls_text(user_index)
        if len(controls_str) > 0:
            self._app_service.text_drawer.add_text(
                controls_str,
                TextOnScreenAlignment.TOP_LEFT,
                destination_mask=Mask.from_index(user_index),
            )

    def sim_update(self, dt: float, post_sim_update_dict):        
        if not self._app_service.hitl_config.experimental.headless.do_headless:
            # Server GUI exit.
            if (
                not self._app_service.hitl_config.networking.enable
                and self._server_gui_input.get_key_down(GuiInput.KeyNS.ESC)
            ):
                self._app_service.end_episode()
                post_sim_update_dict["application_exit"] = True
                return
            
            # Switch the server-controlled user.
            if self._num_agents > 0 and self._server_gui_input.get_key_down(GuiInput.KeyNS.TAB):
                self._server_user_index = (self._server_user_index + 1) % self._num_agents
            
            # Copy server input to user input when server input is active.
            server_user_input = self._user_data[self._server_user_index].gui_input
            if server_user_input.get_any_input():
                self._server_input_enabled = False
            elif self._server_gui_input.get_any_input():
                self._server_input_enabled = True
            if self._server_input_enabled:
                server_user_input.copy_from(self._server_gui_input)

        self._sps_tracker.increment()

        for user_index in self._users.indices(Mask.ALL):
            self._user_data[user_index].update(dt)
            self._update_grasping_and_set_act_hints(user_index)
            self._update_help_text(user_index)
        self._app_service.compute_action_and_step_env()

        # Set the server camera.
        server_cam_transform = self._user_data[
            self._server_user_index
        ].cam_transform
        post_sim_update_dict["cam_transform"] = server_cam_transform

        #  Collect data.
        self._elapsed_time += dt
        if self._is_any_user_active():
            frame_data = self._frame_recorder.record_state(self._elapsed_time, self._user_data)
            self._app_data.session.session_recorder.record_frame(frame_data)

    def _is_any_user_active(self) -> bool:
        for user_index in range(self._app_data.max_user_count):
            if self._user_data[user_index].gui_input.get_any_input() or len(self._user_data[user_index].ui._events) > 0:
                return True
        else:
            return False


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
