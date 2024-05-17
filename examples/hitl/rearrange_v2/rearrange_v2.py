#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# Must call this before importing Habitat or Magnum.
# fmt: off
import ctypes
import sys

sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)
# fmt: on


from typing import List, Optional

# This registers collaboration episodes into this application.
import collaboration_episode_loader  # noqa: 401
import hydra
import magnum as mn
import numpy as np
from ui import UI
from world import World

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
from habitat_hitl.core.user_mask import Mask
from habitat_hitl.environment.camera_helper import CameraHelper
from habitat_hitl.environment.controllers.controller_abc import GuiController
from habitat_hitl.environment.controllers.gui_controller import (
    GuiHumanoidController,
    GuiRobotController,
)
from habitat_hitl.environment.hablab_utils import get_agent_art_obj_transform
from habitat_sim.utils.common import quat_from_magnum, quat_to_coeffs

UP = mn.Vector3(0, 1, 0)


class DataLogger:
    def __init__(self, app_service: AppService):
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


class UserData:
    """
    User-specific states for the ongoing rearrangement session.
    """

    def __init__(
        self,
        app_service: AppService,
        user_index: int,
        world: World,
        gui_agent_controller: GuiController,
        server_sps_tracker: AverageRateTracker,
        client_helper: ClientHelper,
    ):
        self.app_service = app_service
        self.user_index = user_index
        self.gui_agent_controller = gui_agent_controller
        self.server_sps_tracker = server_sps_tracker
        self.client_helper = client_helper
        self.cam_transform = mn.Matrix4.identity_init()
        self.show_gui_text = True
        self.task_instruction = ""
        self.signal_change_episode = False

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
            gui_controller=gui_agent_controller,
            sim=app_service.sim,
            gui_input=self.gui_input,
            gui_drawer=app_service.gui_drawer,
            camera_helper=self.camera_helper,
        )

        # HACK: Work around GuiController input.
        # TODO: Communicate to the controller via action hints.
        gui_agent_controller._gui_input = self.gui_input

    def reset(self):
        self.signal_change_episode = False
        self.camera_helper.update(self._get_camera_lookat_pos(), dt=0)
        self.ui.reset()

    def update(self, dt: float):
        if self.gui_input.get_key_down(GuiInput.KeyNS.H):
            self.show_gui_text = not self.show_gui_text

        if self.gui_input.get_key_down(GuiInput.KeyNS.ZERO):
            self.signal_change_episode = True

        if self.client_helper:
            self.client_helper.update(
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


class AppStateRearrangeV2(AppState):
    """
    Multiplayer rearrangement HITL application.
    """

    def __init__(self, app_service: AppService):
        self._app_service = app_service
        self._gui_agent_controllers = self._app_service.gui_agent_controllers
        self._num_users = len(self._gui_agent_controllers)
        self._can_grasp_place_threshold = (
            self._app_service.hitl_config.can_grasp_place_threshold
        )
        self._num_agents = len(self._gui_agent_controllers)
        self._users = self._app_service.users
        self._paused = False
        self._client_helper: Optional[ClientHelper] = None

        if self._app_service.hitl_config.networking.enable:
            self._client_helper = ClientHelper(
                self._app_service.hitl_config,
                self._app_service.remote_client_state,
                self._app_service.client_message_manager,
                self._users,
            )

        self._server_user_index = 0
        self._server_gui_input = self._app_service.gui_input
        self._server_input_enabled = False

        self._sps_tracker = AverageRateTracker(2.0)

        self._user_data: List[UserData] = []

        self._world = World(self._app_service.sim)

        for user_index in self._users.indices(Mask.ALL):
            self._user_data.append(
                UserData(
                    app_service=app_service,
                    user_index=user_index,
                    world=self._world,
                    gui_agent_controller=self._gui_agent_controllers[
                        user_index
                    ],
                    client_helper=self._client_helper,
                    server_sps_tracker=self._sps_tracker,
                )
            )

    def on_environment_reset(self, episode_recorder_dict):
        self._world.reset()

        # Set the task instruction
        current_episode = self._app_service.env.current_episode

        episode_data = (
            collaboration_episode_loader.load_collaboration_episode_data(
                current_episode
            )
        )
        for user_index in self._users.indices(Mask.ALL):
            self._user_data[
                user_index
            ].task_instruction = episode_data.instruction
            self._user_data[user_index].reset()

        client_message_manager = self._app_service.client_message_manager
        if client_message_manager:
            client_message_manager.signal_scene_change(Mask.ALL)

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

    def _get_gui_controlled_agent_index(self, user_index) -> int:
        return self._gui_agent_controllers[user_index]._agent_idx

    def _get_controls_text(self, user_index: int):
        if self._paused:
            return "Session ended."

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
        if self._paused:
            return ""

        status_str = ""

        if len(self._user_data[user_index].task_instruction) > 0:
            status_str += (
                "Instruction: "
                + self._user_data[user_index].task_instruction
                + "\n"
            )
        if self._user_data[user_index].client_helper and self._user_data[
            user_index
        ].client_helper.do_show_idle_kick_warning(user_index):
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

    def is_user_idle_this_frame(self) -> bool:
        return not self._app_service.gui_input.get_any_input()

    def _check_change_episode(self):
        # If all users signaled to change episode:
        change_episode = True
        for user_index in self._users.indices(Mask.ALL):
            change_episode &= self._user_data[user_index].signal_change_episode

        if (
            change_episode
            and self._app_service.episode_helper.next_episode_exists()
        ):
            # for user_index in self._users.indices(Mask.ALL):
            #    self._user_data[user_index].signal_change_episode = False
            self._app_service.end_episode(do_reset=True)

    def sim_update(self, dt: float, post_sim_update_dict):
        if (
            not self._app_service.hitl_config.networking.enable
            and self._server_gui_input.get_key_down(GuiInput.KeyNS.ESC)
        ):
            self._app_service.end_episode()
            post_sim_update_dict["application_exit"] = True
            return

        # Switch the server-controlled user.
        if (
            self._users.max_user_count > 0
            and self._server_gui_input.get_key_down(GuiInput.KeyNS.TAB)
        ):
            self._server_user_index = (
                self._server_user_index + 1
            ) % self._users.max_user_count

        # Copy server input to user input.
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

        if not self._paused:
            for user_index in self._users.indices(Mask.ALL):
                self._user_data[user_index].update(dt)
                self._update_grasping_and_set_act_hints(user_index)
            self._app_service.compute_action_and_step_env()

        for user_index in self._users.indices(Mask.ALL):
            self._update_help_text(user_index)

        # Set the server camera.
        server_cam_transform = self._user_data[
            self._server_user_index
        ].cam_transform
        post_sim_update_dict["cam_transform"] = server_cam_transform


@hydra.main(
    version_base=None, config_path="config", config_name="rearrange_v2"
)
def main(config):
    if hasattr(config, "habitat_llm") and config.habitat_llm.enable:
        collaboration_episode_loader.register_habitat_llm_extensions(config)
    hitl_main(
        config,
        lambda app_service: AppStateRearrangeV2(app_service),
    )


if __name__ == "__main__":
    register_hydra_plugins()
    main()
