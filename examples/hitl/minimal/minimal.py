#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import magnum

from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.app_states.app_state_abc import AppState
from habitat_hitl.core.hitl_main import hitl_main
from habitat_hitl.core.hydra_utils import register_hydra_plugins
from habitat_hitl.core.key_mapping import KeyCode


class AppStateMinimal(AppState):
    """
    A minimal HITL app that loads and steps a Habitat environment, with
    a fixed overhead camera.
    """

    def __init__(self, app_service: AppService):
        self._app_service = app_service

    def sim_update(self, dt, post_sim_update_dict):
        """
        The HITL framework calls sim_update continuously (for each
        "frame"), before rendering the app's GUI window.
        """
        # run the episode until it ends
        if not self._app_service.env.episode_over:
            self._app_service.compute_action_and_step_env()

        # set the camera for the main 3D viewport
        post_sim_update_dict["cam_transform"] = magnum.Matrix4.look_at(
            eye=magnum.Vector3(-20, 20, -20),
            target=magnum.Vector3(0, 0, 0),
            up=magnum.Vector3(0, 1, 0),
        )

        # exit when the ESC key is pressed
        if self._app_service.gui_input.get_key_down(KeyCode.ESC):
            post_sim_update_dict["application_exit"] = True


@hydra.main(version_base=None, config_path="./", config_name="minimal_cfg")
def main(config):
    hitl_main(config, lambda app_service: AppStateMinimal(app_service))


if __name__ == "__main__":
    register_hydra_plugins()
    main()
