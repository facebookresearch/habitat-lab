#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import hydra
import magnum as mn
import numpy as np
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

class AppData():
    """RearrangeV2 data."""
    def __init__(self, max_user_count: int):
        self.max_user_count = max_user_count
        self.connected_users: Dict[int, ConnectionRecord] = {}

        self.episode_ids: Optional[List[str]] = None
        self.current_episode_index = 0

    def reset(self):
        assert len(self.connected_users) == 0, "Cannot reset RearrangeV2 state if users are still connected!"
        self.episode_ids = None
        self.current_episode_index = 0

class DataLogger:
    def __init__(self, app_service):
        self._app_service = app_service
        self._sim = app_service.sim

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

    def record_state(self, task_completed: bool = False):
        agent_states = self.get_agents_state()
        object_states = self.get_objects_state()

        self._app_service.step_recorder.record("agent_states", agent_states)
        self._app_service.step_recorder.record("object_states", object_states)
        self._app_service.step_recorder.record(
            "task_completed", task_completed
        )

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

    def _kick_all_users(self) -> None:
        "Kick all users."
        self._app_service.remote_client_state.kick(Mask.ALL)

class AppStateMain(AppState):
    """
    Main RearrangeV2 application state.
    It is itself a state machine.

    Flow:
    * 'Clean-up' -> 'Lobby'
    * 'Lobby' -> 'Loading'
    * 'Loading' ->
        * 'Start Screen' if an episode is available.
        * 'End Session' if all episodes are done.
        * 'Clean-up' if a user disconnected or an error occurred.
    * 'Start Screen' -> 'RearrangeV2' or 'Clean-up' (cancel)
    * 'RearrangeV2' -> 'Loading' or 'Clean-up' (cancel)
    * 'End Session' -> 'Upload'
    * 'Upload' -> 'Clean-up'
    """
    def __init__(
            self,
            app_service: AppService,
        ):
            self._app_service = app_service
            self._app_data = AppData(app_service.hitl_config.networking.max_client_count)
            self._app_state: BaseRearrangeState = AppStateCleanup(app_service, self._app_data)

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

        # If a user has disconnected, send a cancellation signal to the current state.
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

    def record_state(self): pass  # Unused.


class AppStateLobby(BaseRearrangeState):
    """
    Idle state.
    Ends when the target user count is reached.
    """
    def __init__(self, app_service: AppService, app_data: AppData,):
        super().__init__(app_service, app_data)
    def get_next_state(self) -> Optional[BaseRearrangeState]:
        if len(self._app_data.connected_users) == self._app_data.max_user_count:
            return AppStateLoadEpisode(self._app_service, self._app_data)
        return None

    def sim_update(self, dt: float, post_sim_update_dict):
        missing_users = self._app_data.max_user_count - len(self._app_data.connected_users)
        s = "s" if missing_users > 1 else ""
        message = f"Waiting for {missing_users} participant{s} to join."
        self._status_message(message)

class AppStateLoadEpisode(BaseRearrangeState):
    """
    Load an episode.
    A loading screen is displaying while the content loads.
    * If no episode set is selected, create a new episode set from connection record.
    * If a next episode exists, fires up RearrangeV2.
    * If all episodes are done, end session.
    Cancellable.
    """
    def __init__(self, app_service: AppService, app_data: AppData,):
        super().__init__(app_service, app_data)
        self._loading = True
        self._session_ended = False
        self._frame_number = 0
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
        # TODO: Clean this up.
        if self._frame_number == 1:
            if self._app_data.episode_ids is None:
                self._load_episode()
            else:
                self._increment_episode()
            self._loading = False

        self._frame_number += 1

    def _load_episode(self):
        data = self._app_data

        # Validate that episodes are selected.
        assert len(data.connected_users) > 0
        connection_record = episodes_str = list(data.connected_users.values())[0]
        if "episodes" not in connection_record:
            print("Users did not request episodes. Cancelling session.")
            self._cancel = True
            return
        episodes_str = connection_record["episodes"]

        # Validate that all users are requesting the same episodes.
        for connection_record in data.connected_users.values():
            if connection_record["episodes"] != episodes_str:
                print("Users are requesting different episodes! Cancelling session.")
                self._cancel = True
                return
            
        # Validate that the episode set is not empty.
        if episodes_str is None or len(episodes_str) == 0:
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
            self._end_session()

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
    def get_next_state(self) -> Optional[BaseRearrangeState]:
        if self._cancel:
            return AppStateCleanup(self._app_service, self._app_data)
        # TODO: If modal dialogue is acknowledged...
        return AppStateRearrangeV2(self._app_service, self._app_data)
        return None

    def sim_update(self, dt: float, post_sim_update_dict):
        # TODO: Top-down view.
        pass # TODO: Modal dialogue.

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
    Actions:
    * Kick all users.
    * Clear data collection output.
    * Clean-up AppData.
    """
    def __init__(self, app_service: AppService, app_data: AppData,):
        super().__init__(app_service, app_data)
    def get_next_state(self) -> Optional[BaseRearrangeState]:
        # TODO: Wait for output folder to be cleaned.

        # Wait for users to be kicked.
        if len(self._app_data.connected_users) == 0:
            # Clean-up AppData
            self._app_data.reset()
            return AppStateLobby(self._app_service, self._app_data)
        else: return None
    def sim_update(self, dt: float, post_sim_update_dict):
        # TODO: Get error message from app_data, and wait a frame until kicking

        self._kick_all_users()

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
        self.gui_input = app_service.remote_client_state.get_gui_input(
            user_index
        )
        self.cam_transform = mn.Matrix4.identity_init()
        self.show_gui_text = True
        self.task_instruction = ""
        self.signal_change_episode = False

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
        self.signal_change_episode = False
        self.camera_helper.update(self._get_camera_lookat_pos(), dt=0)
        self.ui.reset(object_receptacle_pairs)

    def update(self, dt: float):
        if self.gui_input.get_key_down(GuiInput.KeyNS.H):
            self.show_gui_text = not self.show_gui_text

        if self.gui_input.get_key_down(GuiInput.KeyNS.ZERO):
            self.signal_change_episode = True

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

        self._user_data: List[UserData] = []

        self._world = World(app_service.sim)

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
            return AppStateCleanup(self._app_service, self._app_data)

        # Check if all users signaled to terminate episode.
        change_episode = True
        for user_index in self._users.indices(Mask.ALL):
            change_episode &= self._user_data[user_index].signal_change_episode

        # If changing episode, go back to the loading screen.
        # This state takes care of selecting the next episode.
        if change_episode:
            return AppStateLoadEpisode(self._app_service, self._app_data)
        else:
            return None

    def on_environment_reset(self, episode_recorder_dict):
        self._world.reset()

        # Set the task instruction
        current_episode = self._app_service.env.current_episode
        if current_episode.info.get("extra_info") is not None:
            task_instruction = current_episode.info["extra_info"][
                "instruction"
            ]
            # TODO: Users have different instructions.
            for user_index in self._users.indices(Mask.ALL):
                self._user_data[user_index].task_instruction = task_instruction

        client_message_manager = self._app_service.client_message_manager
        if client_message_manager:
            client_message_manager.signal_scene_change(Mask.ALL)

        object_receptacle_pairs = self._create_goal_object_receptacle_pairs()
        for user_index in self._users.indices(Mask.ALL):
            self._user_data[user_index].reset(object_receptacle_pairs)

    def _create_goal_object_receptacle_pairs(
        self,
    ) -> List[Tuple[List[int], List[int]]]:
        """Parse the current episode and returns the goal object-receptacle pairs."""
        sim = self._app_service.sim
        paired_goal_ids: List[Tuple[List[int], List[int]]] = []
        current_episode = self._app_service.env.current_episode
        if current_episode.info.get("extra_info") is not None:
            extra_info = current_episode.info["extra_info"]
            for proposition in extra_info["evaluation_propositions"]:
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
