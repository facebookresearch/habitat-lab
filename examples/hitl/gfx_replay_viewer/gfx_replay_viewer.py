#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hydra
import magnum as mn

from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.app_states.app_state_abc import AppState
from habitat_hitl.core.hitl_main import hitl_main
from habitat_hitl.core.hydra_utils import (
    omegaconf_to_object,
    register_hydra_plugins,
)
from habitat_hitl.core.key_mapping import KeyCode
from habitat_hitl.core.text_drawer import TextOnScreenAlignment
from habitat_hitl.environment.camera_helper import CameraHelper


class AppStateGfxReplayViewer(AppState):
    """
    A minimal HITL app that loads and steps a Habitat environment, with
    a fixed overhead camera.
    """

    def __init__(self, app_service: AppService):
        self._app_service = app_service
        self._sim = app_service.sim

        self._gfx_replay_viewer_cfg = omegaconf_to_object(
            app_service.config.gfx_replay_viewer
        )

        self._player = self._sim.gfx_replay_manager.read_keyframes_from_file(
            self._gfx_replay_viewer_cfg.replay_filepath
        )
        assert self._player

        self._keyframe_index = 0
        self._player.set_keyframe_index(self._keyframe_index)

        self._camera_helper = CameraHelper(
            self._app_service.hitl_config, self._app_service.gui_input
        )
        # not supported for pick_throw_vr
        assert not self._app_service.hitl_config.camera.first_person_mode

        self._camera_lookat_pos = mn.Vector3(0, 0, 0)
        self._camera_helper.update(self._get_camera_lookat_pos(), 0.0)

    def _get_controls_text(self):
        controls_str: str = ""
        controls_str += "ESC: exit\n"
        return controls_str

    def _get_status_text(self):
        status_str = f"keyframe: {self._keyframe_index}\n"
        status_str += f"({self._camera_lookat_pos.x:.1f}, {self._camera_lookat_pos.y:.1f}, {self._camera_lookat_pos.z:.1f})\n"
        return status_str

    def _update_help_text(self):
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

    def _get_camera_lookat_pos(self):
        return self._camera_lookat_pos

    def sim_update(self, dt, post_sim_update_dict):
        gui_input = self._app_service.gui_input
        if gui_input.get_key_down(KeyCode.ESC):
            post_sim_update_dict["application_exit"] = True

        if gui_input.get_key(KeyCode.SPACE):
            self._keyframe_index += 1
            if self._keyframe_index == self._player.get_num_keyframes():
                self._keyframe_index = 0
            self._player.set_keyframe_index(self._keyframe_index)

        xz_forward = self._camera_helper.get_xz_forward()
        if gui_input.get_key(KeyCode.W):
            self._camera_lookat_pos += (
                xz_forward * self._gfx_replay_viewer_cfg.camera_move_speed
            )
        if gui_input.get_key(KeyCode.S):
            self._camera_lookat_pos -= (
                xz_forward * self._gfx_replay_viewer_cfg.camera_move_speed
            )

        self._camera_helper.update(self._get_camera_lookat_pos(), dt)
        self._cam_transform = self._camera_helper.get_cam_transform()
        post_sim_update_dict["cam_transform"] = self._cam_transform

        lookat_ring_radius = 0.1
        lookat_ring_color = mn.Color3(1, 0.75, 0)
        self._app_service.gui_drawer.draw_circle(
            self._camera_lookat_pos,
            lookat_ring_radius,
            lookat_ring_color,
        )

        self._update_help_text()


@hydra.main(
    version_base=None, config_path="./", config_name="gfx_replay_viewer"
)
def main(config):
    hitl_main(config, lambda app_service: AppStateGfxReplayViewer(app_service))


if __name__ == "__main__":
    register_hydra_plugins()
    main()
