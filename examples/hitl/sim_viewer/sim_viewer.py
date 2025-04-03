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
from habitat_hitl.environment.camera_helper import CameraHelper


class AppStateSimViewer(AppState):
    def __init__(
        self,
        app_service: AppService,
    ):
        self._app_service = app_service
        self._gui_input = self._app_service.gui_input

        self._lookat_pos = None
        self._cam_transform = None

        hitl_config = self._app_service.hitl_config
        self._camera_helper = CameraHelper(
            hitl_config, self._app_service.gui_input
        )

        sim_viewer_config = self._app_service.config.sim_viewer
        self._app_service.reconfigure_sim(
            sim_viewer_config.dataset,
            sim_viewer_config.scene,
        )

    def _init_lookat_pos(self):
        self._lookat_pos = mn.Vector3(0.0, 0.0, -1.0)

    def _update_lookat_pos(self):
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

    def _get_camera_lookat_pos(self):
        return self._lookat_pos

    def get_sim(self):
        return self._app_service.sim

    def on_environment_reset(self, episode_recorder_dict):
        self._init_lookat_pos()
        self._camera_helper.update(self._get_camera_lookat_pos(), dt=0)

    def sim_update(self, dt: float, post_sim_update_dict):
        if self._app_service.gui_input.get_key_down(KeyCode.ESC):
            post_sim_update_dict["application_exit"] = True

        self._update_lookat_pos()
        self._camera_helper.update(self._get_camera_lookat_pos(), dt)
        self._cam_transform = self._camera_helper.get_cam_transform()
        post_sim_update_dict["cam_transform"] = self._cam_transform


@hydra.main(version_base=None, config_path="config", config_name="sim_viewer")
def main(config):
    hitl_main(
        config,
        lambda app_service: AppStateSimViewer(app_service),
    )


if __name__ == "__main__":
    register_hydra_plugins()
    main()
