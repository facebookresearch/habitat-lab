#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hydra
import magnum as mn

from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.app_states.app_state_abc import AppState
from habitat_hitl.core.hitl_main import hitl_main
from habitat_hitl.core.hydra_utils import register_hydra_plugins
from habitat_hitl.core.key_mapping import KeyCode
from habitat_hitl.core.text_drawer import TextOnScreenAlignment
from habitat_hitl.environment.camera_helper import CameraHelper


class AppStateBasicViewer(AppState):
    def __init__(
        self,
        app_service: AppService,
    ):
        self._app_service = app_service
        self._gui_input = self._app_service.gui_input

        config = self._app_service.config
        # self._end_on_success = config.habitat.task.end_on_success
        # self._success_measure_name = config.habitat.task.success_measure

        self._lookat_pos = None
        self._cam_transform = None

        self._camera_helper = CameraHelper(
            self._app_service.hitl_config, self._app_service.gui_input
        )
        self._episode_helper = self._app_service.episode_helper
        self._paused = False
        self._do_single_step = False

        self._app_service.reconfigure_sim(
            "data/fpss/hssd-hab-siro.scene_dataset_config.json",
            "102817140.scene_instance.json",
        )

        # Activate the first user.
        self._app_service.users.activate_user(0)

    def _init_lookat_pos(self):
        self._lookat_pos = mn.Vector3(0.0, 0.0, -1.0)

    def _update_lookat_pos(self):
        # update lookat
        move_delta = 0.1
        move = mn.Vector3.zero_init()
        if self._gui_input.get_key(KeyCode.W):
            move.x -= move_delta
        if self._gui_input.get_key(KeyCode.S):
            move.x += move_delta
        if self._gui_input.get_key(KeyCode.E):
            move.y += move_delta
        if self._gui_input.get_key(KeyCode.Q):
            move.y -= move_delta
        if self._gui_input.get_key(KeyCode.J):
            move.z += move_delta
        if self._gui_input.get_key(KeyCode.L):
            move.z -= move_delta

        # align move forward direction with lookat direction
        rot_y_rad = -self._camera_helper.lookat_offset_yaw
        rotation = mn.Quaternion.rotation(
            mn.Rad(rot_y_rad),
            mn.Vector3(0, 1, 0),
        )
        self._lookat_pos += rotation.transform_vector(move)

        # draw lookat point
        radius = 0.15
        self._app_service.gui_drawer.draw_circle(
            self._get_camera_lookat_pos(),
            radius,
            mn.Color3(255 / 255, 0 / 255, 0 / 255),
        )

    @property
    def _env_task_complete(self):
        # return (
        #    self._end_on_success
        #    and self._app_service.get_metrics()[self._success_measure_name]
        # )
        return False

    def _get_camera_lookat_pos(self):
        return self._lookat_pos

    def _get_controls_text(self):
        return
        controls_str: str = ""
        controls_str += "ESC: exit\n"
        if self._episode_helper.next_episode_exists():
            controls_str += "M: next episode\n"
        else:
            controls_str += "no remaining episodes\n"
        if self._env_episode_active():
            controls_str += "P: unpause\n" if self._paused else "P: pause\n"
            controls_str += "Spacebar: single step\n"
        else:
            controls_str += "\n\n"
        controls_str += "R + drag: rotate camera\n"
        controls_str += "Scroll: zoom\n"
        controls_str += "I, K: look up, down\n"
        controls_str += "A, D: turn\n"
        controls_str += "E, Q: move up, down\n"
        controls_str += "W, S: move forward, back\n"

        return controls_str

    def _get_status_text(self):
        return ""
        progress_str = f"episode {self._app_service.episode_helper.current_episode.episode_id}"
        if not self._env_episode_active():
            progress_str += (
                " - task succeeded!"
                if self._env_task_complete
                else " - task ended in failure!"
            )
        elif self._paused:
            progress_str += " - paused"

        # center align the status_str
        max_status_str_len = 50
        status_str = "/n".join(
            line.center(max_status_str_len)
            for line in progress_str.split("/n")
        )

        return status_str

    def _update_help_text(self):
        return
        controls_str = self._get_controls_text()
        if len(controls_str) > 0:
            self._app_service.text_drawer.add_text(
                controls_str, TextOnScreenAlignment.TOP_LEFT
            )

        status_str = self._get_status_text()
        if len(status_str) > 0:
            self._app_service.text_drawer.add_text(
                status_str,
                TextOnScreenAlignment.TOP_CENTER,
                text_delta_x=-120,
            )

    def get_sim(self):
        return self._app_service.sim

    def on_environment_reset(self, episode_recorder_dict):
        self._init_lookat_pos()
        self._camera_helper.update(self._get_camera_lookat_pos(), dt=0)

    def sim_update(self, dt, post_sim_update_dict):
        if self._app_service.gui_input.get_key_down(KeyCode.ESC):
            self._app_service.end_episode()
            post_sim_update_dict["application_exit"] = True

        if self._app_service.gui_input.get_key_down(KeyCode.P):
            self._paused = not self._paused

        if self._app_service.gui_input.get_key_down(KeyCode.SPACE):
            self._do_single_step = True
            self._paused = True

        is_paused_this_frame = self._paused and not self._do_single_step

        if (
            self._app_service.gui_input.get_key_down(KeyCode.M)
            and self._episode_helper.next_episode_exists()
            and not is_paused_this_frame
        ):
            self._app_service.end_episode(do_reset=True)

        self._update_lookat_pos()
        if not is_paused_this_frame:
            # self._app_service.compute_action_and_step_env()
            self._do_single_step = False

        self._camera_helper.update(self._get_camera_lookat_pos(), dt)

        self._cam_transform = self._camera_helper.get_cam_transform()
        post_sim_update_dict["cam_transform"] = self._cam_transform

        self._update_help_text()


@hydra.main(version_base=None, config_path="config", config_name="sim_viewer")
def main(config):
    hitl_main(
        config,
        lambda app_service: AppStateBasicViewer(app_service),
    )


if __name__ == "__main__":
    register_hydra_plugins()
    main()
