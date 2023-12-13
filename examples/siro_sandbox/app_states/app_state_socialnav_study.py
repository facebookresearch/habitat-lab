#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
from app_states.app_state_abc import AppState
from camera_helper import CameraHelper
from controllers.fetch_baselines_controller import FetchState
from gui_rearrange_helper import GuiRearrangeHelper
from utils.hablab_utils import get_agent_art_obj_transform

from utils.gui.gui_input import GuiInput
from utils.gui.text_drawer import TextOnScreenAlignment


class AppStateSocialNavStudy(AppState):
    def __init__(
        self,
        sandbox_service,
        gui_agent_ctrl,
        robot_agent_ctrl,
    ):
        self._sandbox_service = sandbox_service
        self._gui_agent_ctrl = gui_agent_ctrl
        self._state_machine_robot_ctrl = robot_agent_ctrl

        self._cam_transform = None
        self._camera_helper = CameraHelper(
            self._sandbox_service.args, self._sandbox_service.gui_input
        )

        self._rearrange_helper = GuiRearrangeHelper(
            self._sandbox_service,
            self._gui_agent_ctrl,
            self._camera_helper,
        )

        self._episode_helper = self._sandbox_service.episode_helper

    def on_environment_reset(self, episode_recorder_dict):
        self._camera_helper.update(self._get_camera_lookat_pos(), dt=0)
        self._rearrange_helper.on_environment_reset()

        if episode_recorder_dict:
            episode_recorder_dict[
                "target_obj_ids"
            ] = self._rearrange_helper.target_obj_ids
            episode_recorder_dict[
                "goal_positions"
            ] = self._rearrange_helper.goal_positions

    def get_sim(self):
        return self._sandbox_service.sim

    def _get_controls_text(self):
        def get_grasp_release_controls_text():
            if self._rearrange_helper.held_target_obj_idx is not None:
                return "Spacebar: put down\n"
            else:
                return "Spacebar: pick up\n"

        controls_str: str = ""
        controls_str += "ESC: exit\n"
        if self._episode_helper.next_episode_exists():
            controls_str += "M: next episode\n"

        if self._env_episode_active():
            if self._camera_helper.first_person_mode:
                # controls_str += "Left-click: toggle cursor\n"  # make this "unofficial" for now
                controls_str += "I, K: look up, down\n"
                controls_str += "A, D: turn\n"
                controls_str += "W, S: walk\n"
                controls_str += get_grasp_release_controls_text()
            # third-person mode
            else:
                controls_str += "R + drag: rotate camera\n"
                controls_str += "Right-click: walk\n"
                controls_str += "A, D: turn\n"
                controls_str += "W, S: walk\n"
                controls_str += "Scroll: zoom\n"
                controls_str += get_grasp_release_controls_text()

        return controls_str

    @property
    def _env_task_complete(self):
        return (
            self._rearrange_helper.end_on_success
            and self._sandbox_service.get_metrics()[
                self._rearrange_helper.success_measure_name
            ]
        )

    def _env_episode_active(self) -> bool:
        """
        Returns True if current episode is active:
        1) not self._sandbox_service.env.episode_over - none of the constraints is violated, or
        2) not self._env_task_complete - success measure value is not True
        """
        return not (
            self._sandbox_service.env.episode_over or self._env_task_complete
        )

    def _get_status_text(self):
        num_remaining_objects = self._rearrange_helper.num_remaining_objects
        num_busy_objects = self._rearrange_helper.num_busy_objects
        held_target_obj_idx = self._rearrange_helper.held_target_obj_idx

        assert num_remaining_objects is not None
        assert num_busy_objects is not None

        status_str = ""
        if not self._env_episode_active():
            if self._env_task_complete:
                status_str += "Task complete!\n"
            else:
                status_str += "Oops! Something went wrong.\n"
        elif held_target_obj_idx is not None:
            status_str += "Place the object at its goal location!\n"
        elif num_remaining_objects > 0:
            status_str += "Move the remaining {} object{}!".format(
                num_remaining_objects,
                "s" if num_remaining_objects > 1 else "",
            )
        elif num_busy_objects > 0:
            status_str += "Just wait! The robot is moving the last object.\n"
        else:
            # we don't expect to hit this case ever
            status_str += "Oops! Something went wrong.\n"

        # center align the status_str
        max_status_str_len = 50
        status_str = "/n".join(
            line.center(max_status_str_len) for line in status_str.split("/n")
        )

        return status_str

    def _update_help_text(self):
        controls_str = self._get_controls_text()
        if len(controls_str) > 0:
            self._sandbox_service.text_drawer.add_text(
                controls_str, TextOnScreenAlignment.TOP_LEFT
            )

        status_str = self._get_status_text()
        if len(status_str) > 0:
            self._sandbox_service.text_drawer.add_text(
                status_str,
                TextOnScreenAlignment.TOP_CENTER,
                text_delta_x=-280,
                text_delta_y=-50,
            )

        num_episodes_remaining = (
            self._episode_helper.num_iter_episodes
            - self._episode_helper.num_episodes_done
        )
        progress_str = f"{num_episodes_remaining} episodes left"
        self._sandbox_service.text_drawer.add_text(
            progress_str,
            TextOnScreenAlignment.TOP_RIGHT,
            text_delta_x=370,
        )

    def get_num_agents(self):
        return len(self.get_sim().agents_mgr._all_agent_data)

    def record_state(self):
        agent_states = []
        for agent_idx in range(self.get_num_agents()):
            agent_root = get_agent_art_obj_transform(self.get_sim(), agent_idx)
            rotation_quat = mn.Quaternion.from_matrix(agent_root.rotation())
            rotation_list = list(rotation_quat.vector) + [rotation_quat.scalar]
            pos = agent_root.translation

            snap_idx = (
                self.get_sim()
                .agents_mgr._all_agent_data[agent_idx]
                .grasp_mgr.snap_idx
            )

            agent_states.append(
                {
                    "position": pos,
                    "rotation_xyzw": rotation_list,
                    "grasp_mgr_snap_idx": snap_idx,
                }
            )

        self._sandbox_service.step_recorder.record(
            "agent_states", agent_states
        )

        self._sandbox_service.step_recorder.record(
            "target_object_positions",
            self._rearrange_helper.get_target_object_positions(),
        )

    def _get_camera_lookat_pos(self):
        agent_root = get_agent_art_obj_transform(
            self.get_sim(),
            self._rearrange_helper.get_gui_controlled_agent_index(),
        )
        lookat_y_offset = mn.Vector3(0, 1, 0)
        lookat = agent_root.translation + lookat_y_offset
        return lookat

    def _check_update_robot_state(self):
        if self._sandbox_service.gui_input.get_key_down(GuiInput.KeyNS.O):
            current_state = self._state_machine_robot_ctrl.current_state
            if current_state == FetchState.FOLLOW:
                self._state_machine_robot_ctrl.current_state = (
                    FetchState.FOLLOW_ORACLE
                )
            elif current_state == FetchState.FOLLOW_ORACLE:
                self._state_machine_robot_ctrl.current_state = (
                    FetchState.FOLLOW
                )
            else:
                # do nothing
                pass

    def sim_update(self, dt, post_sim_update_dict):
        if self._sandbox_service.gui_input.get_key_down(GuiInput.KeyNS.ESC):
            self._sandbox_service.end_episode()
            post_sim_update_dict["application_exit"] = True

        if (
            self._sandbox_service.gui_input.get_key_down(GuiInput.KeyNS.M)
            and self._episode_helper.next_episode_exists()
        ):
            self._sandbox_service.end_episode(do_reset=True)

        if self._env_episode_active():
            self._rearrange_helper.update()
            self._check_update_robot_state()
            self._sandbox_service.compute_action_and_step_env()

        self._camera_helper.update(self._get_camera_lookat_pos(), dt)

        self._cam_transform = self._camera_helper.get_cam_transform()
        post_sim_update_dict["cam_transform"] = self._cam_transform

        self._update_help_text()

    def is_app_state_done(self):
        # terminal neverending app state
        return False
