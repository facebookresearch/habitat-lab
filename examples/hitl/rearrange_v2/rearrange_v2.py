#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, List, Tuple

import hydra
import magnum as mn
import numpy as np
from ui import UI

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
from habitat_hitl.environment.camera_helper import CameraHelper
from habitat_hitl.environment.controllers.gui_controller import (
    GuiHumanoidController,
    GuiRobotController,
)
from habitat_hitl.environment.hablab_utils import get_agent_art_obj_transform
from habitat_sim.utils.common import quat_from_magnum, quat_to_coeffs


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


class AppStateRearrangeV2(AppState):
    """
    Todo
    """

    def __init__(self, app_service: AppService):
        self._app_service = app_service
        self._gui_agent_controllers = self._app_service.gui_agent_controllers
        self._num_users = len(self._gui_agent_controllers)
        self._can_grasp_place_threshold = (
            self._app_service.hitl_config.can_grasp_place_threshold
        )

        self._sim = app_service.sim
        self._cam_transform = None
        self._camera_user_index = 0
        self._paused = False
        self._show_gui_text = True

        self._camera_helper = CameraHelper(
            self._app_service.hitl_config,
            self._app_service.gui_input,
        )
        self._client_helper = None
        if self._app_service.hitl_config.networking.enable:
            self._client_helper = ClientHelper(self._app_service)

        self._sps_tracker = AverageRateTracker(2.0)

        self._task_instruction = ""
        self._data_logger = DataLogger(app_service=self._app_service)

        self._ui = UI(
            hitl_config=app_service.hitl_config,
            user_index=0,
            gui_controller=self._gui_agent_controllers[0],
            sim=self._sim,
            gui_input=app_service.gui_input,
            gui_drawer=app_service.gui_drawer,
            camera_helper=self._camera_helper,
        )

    # needed to avoid spurious mypy attr-defined errors
    @staticmethod
    def get_sim_utilities() -> Any:
        return sim_utilities

    def on_environment_reset(self, episode_recorder_dict):
        object_receptacle_pairs = self._create_goal_object_receptacle_pairs()
        self._ui.reset(object_receptacle_pairs)

        self._camera_helper.update(self._get_camera_lookat_pos(), dt=0)

        # Set the task instruction
        current_episode = self._app_service.env.current_episode
        if current_episode.info.get("extra_info") is not None:
            self._task_instruction = current_episode.info["extra_info"][
                "instruction"
            ]

        client_message_manager = self._app_service.client_message_manager
        if client_message_manager:
            client_message_manager.signal_scene_change()

    def get_sim(self):
        return self._app_service.sim

    def _create_goal_object_receptacle_pairs(
        self,
    ) -> List[Tuple[List[int], List[int]]]:
        """Parse the current episode and returns the goal object-receptacle pairs."""
        sim = self.get_sim()
        paired_goal_ids: List[Tuple[List[int], List[int]]] = []
        current_episode = self._app_service.env.current_episode
        if current_episode.info.get("extra_info") is not None:
            extra_info = current_episode.info["extra_info"]
            self._task_instruction = extra_info["instruction"]
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

    def _update_grasping_and_set_act_hints(self, user_index):
        gui_agent_controller = self._gui_agent_controllers[user_index]
        assert isinstance(
            gui_agent_controller, (GuiHumanoidController, GuiRobotController)
        )
        gui_agent_controller.set_act_hints(
            walk_dir=None,
            distance_multiplier=1.0,
            grasp_obj_idx=None,
            do_drop=None,
            cam_yaw=self._camera_helper.lookat_offset_yaw,
            throw_vel=None,
            reach_pos=None,
        )

    def get_gui_controlled_agent_index(self, user_index):
        return self._gui_agent_controllers[user_index]._agent_idx

    def _get_controls_text(self):
        if self._paused:
            return "Session ended."

        if not self._show_gui_text:
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

    def _get_status_text(self):
        if self._paused:
            return ""

        status_str = ""

        if len(self._task_instruction) > 0:
            status_str += "Instruction: " + self._task_instruction + "\n"
        if (
            self._client_helper
            and self._client_helper.do_show_idle_kick_warning
        ):
            status_str += (
                "\n\nAre you still there?\nPress any key to keep playing!\n"
            )

        return status_str

    def _update_help_text(self):
        status_str = self._get_status_text()
        if len(status_str) > 0:
            self._app_service.text_drawer.add_text(
                status_str,
                TextOnScreenAlignment.TOP_CENTER,
                text_delta_x=-280,
                text_delta_y=-50,
            )

        controls_str = self._get_controls_text()
        if len(controls_str) > 0:
            self._app_service.text_drawer.add_text(
                controls_str, TextOnScreenAlignment.TOP_LEFT
            )

    def _get_camera_lookat_pos(self):
        agent_root = get_agent_art_obj_transform(
            self.get_sim(),
            self.get_gui_controlled_agent_index(self._camera_user_index),
        )
        lookat_y_offset = mn.Vector3(0, 1, 0)
        lookat = agent_root.translation + lookat_y_offset
        return lookat

    def is_user_idle_this_frame(self):
        return not self._app_service.gui_input.get_any_key_down()

    def _check_change_episode(self):
        if self._paused or not self._app_service.gui_input.get_key_down(
            GuiInput.KeyNS.ZERO
        ):
            return

        if self._app_service.episode_helper.next_episode_exists():
            self._app_service.end_episode(do_reset=True)

    def sim_update(self, dt, post_sim_update_dict):
        if (
            not self._app_service.hitl_config.networking.enable
            and self._app_service.gui_input.get_key_down(GuiInput.KeyNS.ESC)
        ):
            self._app_service.end_episode()
            post_sim_update_dict["application_exit"] = True
            return

        self._sps_tracker.increment()

        if self._client_helper:
            self._client_helper.update(
                self.is_user_idle_this_frame(),
                self._sps_tracker.get_smoothed_rate(),
            )

        if self._app_service.gui_input.get_key_down(GuiInput.KeyNS.H):
            self._show_gui_text = not self._show_gui_text

        self._check_change_episode()

        if not self._paused:
            for user_index in range(self._num_users):
                self._ui.update()  # TODO: One UI per user.
                self._update_grasping_and_set_act_hints(user_index)
            self._app_service.compute_action_and_step_env()
        else:
            # temp hack: manually add a keyframe while paused
            self.get_sim().gfx_replay_manager.save_keyframe()

        self._camera_helper.update(self._get_camera_lookat_pos(), dt)

        self._cam_transform = self._camera_helper.get_cam_transform()
        post_sim_update_dict["cam_transform"] = self._cam_transform

        self._ui.draw_ui()  # TODO: One UI per user.
        self._update_help_text()

    def record_state(self):
        task_completed = self._app_service.gui_input.get_key_down(
            GuiInput.KeyNS.ZERO
        )
        self._data_logger.record_state(task_completed=task_completed)


@hydra.main(
    version_base=None, config_path="config", config_name="rearrange_v2"
)
def main(config):
    hitl_main(
        config,
        lambda app_service: AppStateRearrangeV2(app_service),
    )


if __name__ == "__main__":
    register_hydra_plugins()
    main()
