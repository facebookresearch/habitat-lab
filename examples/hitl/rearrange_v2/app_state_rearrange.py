#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import magnum as mn
import numpy as np
from app_data import AppData
from app_state_base import AppStateBase
from app_states import (
    create_app_state_end_session,
    create_app_state_load_episode,
)
from ui import UI
from util import UP
from world import World

from habitat.sims.habitat_simulator import sim_utilities
from habitat_hitl._internal.networking.average_rate_tracker import (
    AverageRateTracker,
)
from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.core.client_message_manager import UIButton
from habitat_hitl.core.gui_input import GuiInput
from habitat_hitl.core.text_drawer import TextOnScreenAlignment
from habitat_hitl.core.user_mask import Mask, Users
from habitat_hitl.environment.camera_helper import CameraHelper
from habitat_hitl.environment.controllers.controller_abc import GuiController
from habitat_hitl.environment.controllers.gui_controller import (
    GuiHumanoidController,
    GuiRobotController,
)
from habitat_hitl.environment.hablab_utils import get_agent_art_obj_transform
from habitat_sim.utils.common import quat_from_magnum, quat_to_coeffs


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
        self, elapsed_time: float, user_data: List[UserData]
    ) -> Dict[str, Any]:
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
                "events": list(u.ui._events),
            }
            data["users"].append(user_data_dict)
            u.ui._events.clear()  # Sloppy

        return data


from session import Session


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
        self.gui_input: GuiInput = (
            app_service.remote_client_state.get_gui_input(user_index)
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
                self.cam_transform,
                destination_mask=Mask.from_index(self.user_index),
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


class AppStateRearrangeV2(AppStateBase):
    """
    Todo
    """

    def __init__(
        self, app_service: AppService, app_data: AppData, session: Session
    ):
        super().__init__(app_service, app_data)
        self._session = session
        self._auto_save_keyframes = False  # Done in env step

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

        self._frame_recorder = FrameRecorder(
            app_service, app_data, self._world
        )

        for user_index in self._users.indices(Mask.ALL):
            self._user_data.append(
                UserData(
                    app_service=app_service,
                    user_index=user_index,
                    world=self._world,
                    gui_agent_controller=self._gui_agent_controllers[
                        user_index
                    ],
                    server_sps_tracker=self._sps_tracker,
                )
            )

        # Reset the environment immediately.
        self.on_environment_reset(None)

    def get_next_state(self) -> Optional[AppStateBase]:
        # If cancelled, skip upload and clean-up.
        if self._cancel:
            return create_app_state_end_session(
                self._app_service, self._app_data, self._session
            )

        # Check if all users signaled to terminate episode.
        change_episode = self._is_episode_finished()

        # If changing episode, go back to the loading screen.
        # This state takes care of selecting the next episode.
        if change_episode:
            return create_app_state_load_episode(
                self._app_service, self._app_data, self._session
            )
        else:
            return None

    def on_enter(self):
        super().on_enter()
        episode = self._app_service.episode_helper.current_episode
        self._session.session_recorder.start_episode(
            episode.episode_id,
            episode.scene_id,
            episode.scene_dataset_config,
        )

    def on_exit(self):
        super().on_exit()

        self._session.session_recorder.end_episode(
            success=self._is_episode_finished()
        )

    def _is_episode_finished(self) -> bool:
        """
        Determines whether all users have finished their tasks.
        """
        return all(
            self._user_data[user_index].task_completed
            for user_index in self._users.indices(Mask.ALL)
        )

    def on_environment_reset(self, episode_recorder_dict):
        self._world.reset()

        # Set the task instruction
        current_episode = self._app_service.env.current_episode
        if hasattr(current_episode, "instruction"):
            task_instruction = current_episode.instruction
            # TODO: Users will have different instructions.
            for user_index in self._users.indices(Mask.ALL):
                self._user_data[user_index].task_instruction = task_instruction

        object_receptacle_pairs = self._create_goal_object_receptacle_pairs()
        for user_index in self._users.indices(Mask.ALL):
            self._user_data[user_index].reset(object_receptacle_pairs)

        # Insert a keyframe immediately.
        self._app_service.sim.gfx_replay_manager.save_keyframe()

    def _create_goal_object_receptacle_pairs(
        self,
    ) -> List[Tuple[List[int], List[int]]]:
        """Parse the current episode and returns the goal object-receptacle pairs."""
        sim = self._app_service.sim
        paired_goal_ids: List[Tuple[List[int], List[int]]] = []
        current_episode = self._app_service.env.current_episode
        # Typing unavailable for this episode type.
        if hasattr(current_episode, "evaluation_propositions"):
            evaluation_propositions = current_episode.evaluation_propositions
            for proposition in evaluation_propositions:
                object_ids: List[int] = []
                if hasattr(proposition, "args"):
                    if "object_handles" in proposition.args:
                        object_handles = proposition.args["object_handles"]
                        for object_handle in object_handles:
                            obj = sim_utilities.get_obj_from_handle(
                                sim, object_handle
                            )
                            object_id = obj.object_id
                            object_ids.append(object_id)
                        receptacle_ids: List[int] = []
                    if "receptacle_handles" in proposition.args:
                        receptacle_handles = proposition.args[
                            "receptacle_handles"
                        ]
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
        controls_str += f"Episode: {self._app_data.current_episode_index}\n"
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
        if self._app_service.remote_client_state._client_helper.do_show_idle_kick_warning(
            user_index
        ):
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
            cancel = self._app_service.remote_client_state.ui_button_clicked(
                user_index, ui_button_id
            )
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
            if self._num_agents > 0 and self._server_gui_input.get_key_down(
                GuiInput.KeyNS.TAB
            ):
                self._server_user_index = (
                    self._server_user_index + 1
                ) % self._num_agents

            # Copy server input to user input when server input is active.
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
        self._app_service.compute_action_and_step_env()

        # Set the server camera.
        server_cam_transform = self._user_data[
            self._server_user_index
        ].cam_transform
        post_sim_update_dict["cam_transform"] = server_cam_transform

        #  Collect data.
        self._elapsed_time += dt
        if self._is_any_user_active():
            # TODO: Refactor
            frame_data = self._frame_recorder.record_state(
                self._elapsed_time, self._user_data
            )
            self._session.session_recorder.record_frame(frame_data)

    def _is_any_user_active(self) -> bool:
        return any(
            self._user_data[user_index].gui_input.get_any_input()
            or len(self._user_data[user_index].ui._events) > 0
            for user_index in range(self._app_data.max_user_count)
        )
