#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

import magnum as mn
import numpy as np
from app_data import AppData
from app_state_base import AppStateBase
from app_states import (
    create_app_state_cancel_session,
    create_app_state_load_episode,
)
from end_episode_form import EndEpisodeForm, ErrorReport
from metrics import Metrics
from session import Session
from ui import UI
from util import UP
from world import World

from habitat_hitl._internal.networking.average_rate_tracker import (
    AverageRateTracker,
)
from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.core.key_mapping import KeyCode
from habitat_hitl.core.text_drawer import TextOnScreenAlignment
from habitat_hitl.core.user_mask import Mask, Users
from habitat_hitl.environment.camera_helper import CameraHelper
from habitat_hitl.environment.controllers.controller_abc import (
    Controller,
    GuiController,
)
from habitat_hitl.environment.controllers.gui_controller import (
    GuiHumanoidController,
    GuiRobotController,
)
from habitat_hitl.environment.hablab_utils import get_agent_art_obj_transform
from habitat_sim.utils.common import quat_from_magnum, quat_to_coeffs

PIP_VIEWPORT_ID = 0  # ID of the picture-in-picture viewport that shows other agent's perspective.


class EpisodeCompletionStatus(Enum):
    PENDING = (0,)
    SUCCESS = (1,)
    FAILURE = (2,)


class FrameRecorder:
    def __init__(
        self, app_service: AppService, app_data: AppData, world: World
    ):
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

    def record_state(
        self,
        elapsed_time: float,
        user_data: List[UserData],
        task_percent_complete: Optional[float],
    ) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "t": elapsed_time,
            "users": [],
            "object_states": self.get_objects_state(),
            "agent_states": self.get_agents_state(),
            "task_percent_complete": task_percent_complete,
        }

        for user_index in range(len(user_data)):
            u = user_data[user_index]
            user_data_dict = {
                "finished_episode": u.agent_data.episode_completion_status
                != EpisodeCompletionStatus.PENDING,
                "camera_transform": u.agent_data.cam_transform,
                "held_object": u.ui._held_object_id,
                "hovered_object": u.ui._hover_selection.object_id,
                "events": u.pop_ui_events(),
            }
            data["users"].append(user_data_dict)

        return data


class AgentData:
    """
    Agent-specific states for the ongoing rearrangement session.
    Agents can be controlled by either a user or an AI.
    """

    def __init__(
        self,
        app_service: AppService,
        world: World,
        agent_controller: Controller,
        agent_index: int,
        render_camera: Optional[Any],
    ):
        self.app_service = app_service
        self.world = world
        self.agent_controller = agent_controller
        self.agent_index = agent_index

        self.task_instruction = ""

        self.render_camera = render_camera
        self.cam_transform = mn.Matrix4.identity_init()

        self.episode_completion_status = EpisodeCompletionStatus.PENDING

    def update_camera_from_sensor(self) -> None:
        """
        Update the camera transform from the agent's sensor.
        Agents controlled by users have their camera updated using CameraHelper.
        For AI-controlled agents, the camera transform can be inferred from this function.
        """
        if self.render_camera is not None:
            self.cam_transform = np.linalg.inv(
                self.render_camera.camera_matrix
            )


class UserData:
    """
    User-specific states for the ongoing rearrangement session.
    """

    def __init__(
        self,
        app_service: AppService,
        user_index: int,
        world: World,
        agent_data: AgentData,
        server_sps_tracker: AverageRateTracker,
    ):
        self.app_service = app_service
        self.user_index = user_index
        self.world = world
        self.agent_data = agent_data
        self.server_sps_tracker = server_sps_tracker
        self.client_helper = (
            self.app_service.remote_client_state._client_helper
        )
        self.show_gui_text = True
        self.pip_initialized = False

        gui_agent_controller = agent_data.agent_controller
        assert isinstance(
            gui_agent_controller, GuiController
        ), "User agent controller must be a GuiController"
        self.gui_agent_controller = gui_agent_controller

        # Events for data collection.
        self.ui_events: List[Dict[str, Any]] = []

        # If in remote mode, get the remote input. Else get the server (local) input.
        self.gui_input = (
            app_service.remote_client_state.get_gui_input(user_index)
            if app_service.remote_client_state is not None
            else self.app_service.gui_input
        )

        self.camera_helper = CameraHelper(
            app_service.hitl_config,
            self.gui_input,
        )

        self.ui = UI(
            hitl_config=app_service.hitl_config,
            user_index=user_index,
            world=world,
            gui_controller=self.gui_agent_controller,
            sim=app_service.sim,
            gui_input=self.gui_input,
            gui_drawer=app_service.gui_drawer,
            camera_helper=self.camera_helper,
        )

        self.end_episode_form = EndEpisodeForm(user_index, app_service)

        # Register UI callbacks
        self.ui.on_pick.registerCallback(self._on_pick)
        self.ui.on_place.registerCallback(self._on_place)
        self.ui.on_open.registerCallback(self._on_open)
        self.ui.on_close.registerCallback(self._on_close)

        self.end_episode_form.on_cancel.registerCallback(
            self._on_episode_form_cancelled
        )
        self.end_episode_form.on_episode_success.registerCallback(
            self._on_episode_finished
        )
        self.end_episode_form.on_error_reported.registerCallback(
            self._on_error_reported
        )

        # HACK: Work around GuiController input.
        # TODO: Communicate to the controller via action hints.
        gui_agent_controller._gui_input = self.gui_input

    def reset(self):
        self.camera_helper.update(self._get_camera_lookat_pos(), dt=0)
        self.ui.reset()

        # If networking is enabled...
        if self.app_service.client_message_manager:
            # Assign user agent objects to their own layer.
            agent_index = self.gui_agent_controller._agent_idx
            agent_object_ids = self.world.get_agent_object_ids(agent_index)
            for agent_object_id in agent_object_ids:
                self.app_service.client_message_manager.set_object_visibility_layer(
                    object_id=agent_object_id,
                    layer_id=agent_index,
                    destination_mask=Mask.from_index(self.user_index),
                )

            # Show all layers except "user_index" in the default viewport.
            # This hides the user's own agent in the first person view.
            self.app_service.client_message_manager.set_viewport_properties(
                viewport_id=-1,
                visible_layer_ids=Mask.all_except_index(agent_index),
                destination_mask=Mask.from_index(self.user_index),
            )

    def update(self, dt: float):
        if self.end_episode_form.is_form_shown():
            self.end_episode_form.step()
            return

        if self.gui_input.get_key_down(KeyCode.H):
            self.show_gui_text = not self.show_gui_text

        if self.gui_input.get_key_down(KeyCode.ZERO):
            self.end_episode_form.show()

        if self.client_helper:
            self.client_helper.update(
                self.user_index,
                self._is_user_idle_this_frame(),
                self.server_sps_tracker.get_smoothed_rate(),
            )

        self.camera_helper.update(self._get_camera_lookat_pos(), dt)
        self.agent_data.cam_transform = self.camera_helper.get_cam_transform()

        if self.app_service.hitl_config.networking.enable:
            self.app_service._client_message_manager.update_camera_transform(
                self.agent_data.cam_transform,
                destination_mask=Mask.from_index(self.user_index),
            )

        self.ui.update()
        self.ui.draw_ui()

    def draw_pip_viewport(self, pip_agent_data: AgentData):
        """
        Draw a picture-in-picture viewport showing another agent's perspective.
        """
        # If networking is disabled, skip.
        if not self.app_service.client_message_manager:
            return

        # Lazy init:
        if not self.pip_initialized:
            self.pip_initialized = True

            # Assign pip agent objects to their own layer.
            pip_agent_index = pip_agent_data.agent_index
            agent_object_ids = self.world.get_agent_object_ids(pip_agent_index)
            for agent_object_id in agent_object_ids:
                self.app_service.client_message_manager.set_object_visibility_layer(
                    object_id=agent_object_id,
                    layer_id=pip_agent_index,
                    destination_mask=Mask.from_index(self.user_index),
                )

            # Define picture-in-picture (PIP) viewport.
            # Show all layers except "pip_user_index".
            # This hides the other agent in the picture-in-picture viewport.
            self.app_service.client_message_manager.set_viewport_properties(
                viewport_id=PIP_VIEWPORT_ID,
                viewport_rect_xywh=[0.8, 0.02, 0.18, 0.18],
                visible_layer_ids=Mask.all_except_index(pip_agent_index),
                destination_mask=Mask.from_index(self.user_index),
            )

        # Show picture-in-picture (PIP) viewport.
        self.app_service.client_message_manager.update_camera_transform(
            cam_transform=pip_agent_data.cam_transform,
            viewport_id=PIP_VIEWPORT_ID,
            destination_mask=Mask.from_index(self.user_index),
        )

    def pop_ui_events(self) -> List[Dict[str, Any]]:
        events = list(self.ui_events)
        self.ui_events.clear()
        return events

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

    def _on_pick(self, e: UI.PickEventData):
        self.ui_events.append(
            {
                "type": "pick",
                "obj_handle": e.object_handle,
                "obj_id": e.object_id,
            }
        )

    def _on_place(self, e: UI.PlaceEventData):
        self.ui_events.append(
            {
                "type": "place",
                "obj_handle": e.object_handle,
                "obj_id": e.object_id,
                "receptacle_id": e.receptacle_id,
            }
        )

    def _on_open(self, e: UI.OpenEventData):
        self.ui_events.append(
            {
                "type": "open",
                "obj_handle": e.object_handle,
                "obj_id": e.object_id,
            }
        )

    def _on_close(self, e: UI.CloseEventData):
        self.ui_events.append(
            {
                "type": "close",
                "obj_handle": e.object_handle,
                "obj_id": e.object_id,
            }
        )

    def _on_episode_form_cancelled(self, _e: Any = None):
        self.ui_events.append(
            {
                "type": "end_episode_form_cancelled",
            }
        )
        self.agent_data.episode_completion_status = (
            EpisodeCompletionStatus.PENDING
        )

    def _on_episode_finished(self, _e: Any = None):
        self.ui_events.append(
            {
                "type": "episode_finished",
            }
        )
        self.agent_data.episode_completion_status = (
            EpisodeCompletionStatus.SUCCESS
        )
        print(f"User {self.user_index} has signaled the episode as completed.")

    def _on_error_reported(self, error_report: ErrorReport):
        self.ui_events.append(
            {
                "type": "error_reported",
                "error_report": error_report.user_message,
            }
        )
        self.agent_data.episode_completion_status = (
            EpisodeCompletionStatus.FAILURE
        )
        print(
            f"User {self.user_index} has signaled a problem with the episode: '{error_report.user_message}'."
        )


class AppStateRearrangeV2(AppStateBase):
    """
    Multiplayer rearrangement HITL application.
    """

    def __init__(
        self, app_service: AppService, app_data: AppData, session: Session
    ):
        super().__init__(app_service, app_data)
        sim = app_service.sim
        agent_mgr = sim.agents_mgr
        self._save_keyframes = False  # Done on env step (rearrange_sim).

        self._app_service = app_service
        self._session = session
        self._gui_agent_controllers = app_service.gui_agent_controllers

        self._users = app_service.users
        self._num_users = self._users.max_user_count
        self._agents = Users(
            len(agent_mgr._all_agent_data), activate_users=True
        )
        self._num_agents = self._agents.max_user_count

        self._sps_tracker = AverageRateTracker(2.0)
        self._server_user_index = 0
        self._server_gui_input = app_service.gui_input
        self._server_input_enabled = False
        self._elapsed_time = 0.0

        self._world = World(app_service.sim, app_service.config)
        self._metrics = Metrics(app_service)

        self._agent_to_user_index: Dict[int, int] = {}
        self._user_to_agent_index: Dict[int, int] = {}

        self._agent_data: List[AgentData] = []
        for agent_index in range(self._num_agents):
            agent = agent_mgr._all_agent_data[agent_index]
            camera_name: Optional[Any] = (
                agent.articulated_agent._cameras[0]
                if len(agent.articulated_agent._cameras) > 0
                else None
            )
            render_camera: Optional[Any] = (
                sim.agents[agent_index]._sensors[camera_name].render_camera
                if camera_name is not None
                else None
            )
            agent_controller = app_service.all_agent_controllers[agent_index]

            # Match agent and user indices.
            for user_index in range(len(self._gui_agent_controllers)):
                gui_agent_controller = self._gui_agent_controllers[user_index]
                if gui_agent_controller._agent_idx == agent_index:
                    self._agent_to_user_index[agent_index] = user_index
                    self._user_to_agent_index[user_index] = agent_index
                    break

            self._agent_data.append(
                AgentData(
                    app_service=app_service,
                    world=self._world,
                    agent_controller=agent_controller,
                    agent_index=agent_index,
                    render_camera=render_camera,
                )
            )

        self._user_data: List[UserData] = []
        for user_index in range(self._users.max_user_count):
            agent_data = self._agent_data[
                self._user_to_agent_index[user_index]
            ]
            self._user_data.append(
                UserData(
                    app_service=app_service,
                    user_index=user_index,
                    world=self._world,
                    agent_data=agent_data,
                    server_sps_tracker=self._sps_tracker,
                )
            )

        self._frame_recorder = FrameRecorder(
            app_service, app_data, self._world
        )

        # Reset the environment immediately.
        self.on_environment_reset(None)

    def get_next_state(self) -> Optional[AppStateBase]:
        if self._cancel:
            return create_app_state_cancel_session(
                self._app_service,
                self._app_data,
                self._session,
                "User disconnected",
            )
        elif self._is_episode_finished():
            return create_app_state_load_episode(
                self._app_service, self._app_data, self._session
            )
        else:
            return None

    def on_enter(self):
        super().on_enter()

        user_index_to_agent_index_map: Dict[int, int] = {}
        for user_index in range(len(self._user_data)):
            user_index_to_agent_index_map[user_index] = self._user_data[
                user_index
            ].gui_agent_controller._agent_idx

        episode = self._app_service.episode_helper.current_episode
        self._session.session_recorder.start_episode(
            episode.episode_id,
            episode.scene_id,
            episode.scene_dataset_config,
            user_index_to_agent_index_map,
        )

    def on_exit(self):
        super().on_exit()

        task_percent_complete = self._metrics.get_task_percent_complete()
        feedback = self._metrics.get_task_explanation()

        episode_finished = self._is_episode_finished() and not self._cancel

        self._session.session_recorder.end_episode(
            episode_finished=episode_finished,
            task_percent_complete=task_percent_complete
            if task_percent_complete is not None
            else 1.0,
            task_explanation=feedback,
        )

    def on_environment_reset(self, episode_recorder_dict):
        self._world.reset()

        # Reset AFK timers.
        # TODO: Move to idle_kick_timer class. Make it per-user. Couple it with "user_data" class
        self._app_service.remote_client_state._client_helper.activate_users()

        # Set the task instruction
        current_episode = self._app_service.env.current_episode
        if hasattr(current_episode, "instruction"):
            task_instruction = current_episode.instruction
            # TODO: Agents will have different instructions.
            for agent_index in self._agents.indices(Mask.ALL):
                self._agent_data[
                    agent_index
                ].task_instruction = task_instruction

        for user_index in self._users.indices(Mask.ALL):
            self._user_data[user_index].reset()

        # Insert a keyframe immediately.
        self._app_service.sim.gfx_replay_manager.save_keyframe()

    def _update_grasping_and_set_act_hints(self, user_index: int):
        # TODO: Read/write from grasp manager.
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

    def _get_controls_text(self, user_index: int):
        controls_str: str = ""
        if self._user_data[user_index].show_gui_text:
            controls_str += "H: Toggle help\n"
            controls_str += "Look: Middle click (drag), I, K\n"
            controls_str += "Walk: W, S\n"
            controls_str += "Turn: A, D\n"
            controls_str += "Finish episode: Zero (0)\n"
            controls_str += "Open/close: Double-click\n"
            controls_str += "Pick object: Double-click\n"
            controls_str += "Place object: Right click (hold)\n"

        client_helper = self._app_service.remote_client_state._client_helper
        idle_time = client_helper.get_idle_time(user_index)
        if idle_time > 10:
            controls_str += f"Idle time {idle_time}s\n"

        return controls_str

    def _get_status_text(self, user_index: int):
        status_str = ""

        task_instruction = self._user_data[
            user_index
        ].agent_data.task_instruction
        if len(task_instruction) > 0:
            status_str += "Instruction: " + task_instruction + "\n"

        if (
            self._users.max_user_count > 1
            and self._user_data[
                user_index
            ].agent_data.episode_completion_status
            == EpisodeCompletionStatus.PENDING
        ):
            if self._has_any_agent_finished_success():
                status_str += "\n\nThe other participant signaled that the task is completed.\nPress '0' when you are done."
            elif self._has_any_agent_finished_failure():
                status_str += "\n\nThe other participant signaled a problem with the task.\nPress '0' to continue."

        client_helper = self._app_service.remote_client_state._client_helper
        if client_helper.do_show_idle_kick_warning(user_index):
            remaining_time = str(
                client_helper.get_remaining_idle_time(user_index)
            )
            status_str += f"\n\nAre you still there?\nPress any key in the next {remaining_time}s to keep playing!\n"

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
        if self._is_server_gui_enabled():
            # Server GUI exit.
            if (
                not self._app_service.hitl_config.networking.enable
                and self._server_gui_input.get_key_down(KeyCode.ESC)
            ):
                self._app_service.end_episode()
                post_sim_update_dict["application_exit"] = True
                return

            # Skip the form when changing the episode from the server.
            if self._server_gui_input.get_key_down(KeyCode.ZERO):
                server_user = self._user_data[self._server_user_index]
                if (
                    server_user.agent_data.episode_completion_status
                    == EpisodeCompletionStatus.PENDING
                ):
                    server_user._on_episode_finished()

            # Switch the server-controlled user.
            if self._num_users > 0 and self._server_gui_input.get_key_down(
                KeyCode.TAB
            ):
                self._server_user_index = (
                    self._server_user_index + 1
                ) % self._num_users

        # Update world.
        self._world.update(dt)

        # Copy server input to user input when server input is active.
        if self._app_service.hitl_config.networking.enable:
            server_user_input = self._user_data[
                self._server_user_index
            ].gui_input
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

        # Draw the picture-in-picture showing other agent's perspective.
        if self._num_agents == 2:
            for user_index in range(self._num_users):
                user_agent_idx = self._user_to_agent_index[user_index]
                other_agent_idx = user_agent_idx ^ 1
                other_agent_data = self._agent_data[other_agent_idx]

                # If the other agent is AI-controlled, update its camera.
                if other_agent_idx not in self._user_to_agent_index:
                    other_agent_data.update_camera_from_sensor()

                self._user_data[user_index].draw_pip_viewport(other_agent_data)

        self._app_service.compute_action_and_step_env()

        # Set the server camera.
        server_cam_transform = self._user_data[
            self._server_user_index
        ].agent_data.cam_transform
        post_sim_update_dict["cam_transform"] = server_cam_transform

        #  Collect data.
        self._elapsed_time += dt
        if self._is_any_agent_policy_driven() or self._is_any_user_active():
            frame_data = self._frame_recorder.record_state(
                self._elapsed_time,
                self._user_data,
                self._metrics.get_task_percent_complete(),
            )
            self._session.session_recorder.record_frame(frame_data)

    def _is_any_agent_policy_driven(self) -> bool:
        """
        Returns true if any of the agents is policy-driven.
        Returns false if all agents are user-driven.
        """
        return self._num_agents > self._num_users

    def _is_any_user_active(self) -> bool:
        """
        Returns true if any user is active during the frame.
        """
        return self._is_any_agent_policy_driven() or any(
            self._user_data[user_index].gui_input.get_any_input()
            or len(self._user_data[user_index].ui_events) > 0
            for user_index in range(self._app_data.max_user_count)
        )

    def _has_any_agent_finished_success(self) -> bool:
        """
        Returns true if any agent completed the episode successfully.
        """
        return any(
            self._agent_data[agent_index].episode_completion_status
            == EpisodeCompletionStatus.SUCCESS
            for agent_index in range(self._num_agents)
        )

    def _has_any_agent_finished_failure(self) -> bool:
        """
        Returns true if any agent completed the episode unsuccessfully.
        """
        return any(
            self._agent_data[agent_index].episode_completion_status
            == EpisodeCompletionStatus.FAILURE
            for agent_index in range(self._num_agents)
        )

    def _is_episode_finished(self) -> bool:
        """
        Returns true if all agents finished the episode, regardless of success.
        """
        return all(
            self._agent_data[agent_index].episode_completion_status
            != EpisodeCompletionStatus.PENDING
            for agent_index in range(self._num_agents)
        )

    def _is_episode_successful(self) -> bool:
        """
        Returns true if:
        * 'task_percent_complete' is 100%.
        * All agents finished the episode without reporting an error.
        """

        task_percent_complete = self._metrics.get_task_percent_complete()
        task_successful = (
            # We avoid comparing to 1.0 in case the implementation doesn't return a whole number.
            task_percent_complete > 0.99
            if task_percent_complete is not None
            # If the task success metric isn't available, assume success.
            else True
        )

        all_agents_reported_success = all(
            self._agent_data[agent_index].episode_completion_status
            == EpisodeCompletionStatus.SUCCESS
            for agent_index in range(self._num_agents)
        )

        return task_successful and all_agents_reported_success
