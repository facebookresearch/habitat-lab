#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import hydra
import magnum as mn

from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.app_states.app_state_abc import AppState
from habitat_hitl.core.hitl_main import hitl_main
from habitat_hitl.core.hydra_utils import register_hydra_plugins
from habitat_hitl.core.key_mapping import KeyCode, XRButton
from habitat_hitl.core.text_drawer import TextOnScreenAlignment


class AppStateXRReader(AppState):
    def __init__(
        self,
        app_service: AppService,
    ):
        self._app_service = app_service
        self._gui_input = self._app_service.gui_input
        self._xr_input = self._app_service.remote_client_state.get_xr_input()
        self._app_service.users.activate_user(0)

    def get_sim(self):
        return self._app_service.sim

    def on_environment_reset(self, episode_recorder_dict):
        pass

    def sim_update(self, dt, post_sim_update_dict):
        if self._app_service.gui_input.get_key_down(KeyCode.ESC):
            self._app_service.end_episode()
            post_sim_update_dict["application_exit"] = True
        post_sim_update_dict["cam_transform"] = mn.Matrix4.identity_init()

        left = self._xr_input.left_controller
        right = self._xr_input.right_controller
        state = self._app_service.remote_client_state

        def btn(button_list: list[XRButton]) -> str:
            button_names = [
                button.name
                for button in button_list
                if isinstance(button, XRButton)
            ]
            return f"[{', '.join(button_names)}]"

        def pos(
            legacy_tuple: Optional[tuple[mn.Vector3, mn.Quaternion]]
        ) -> str:
            if legacy_tuple is None:
                return "None"
            v = legacy_tuple[0]
            if v is None:
                return "None"
            return f"{f'{v.x:.2f}'}, {f'{v.y:.2f}'}, {f'{v.z:.2f}'}"

        text = f"""
        Left Controller:
        - Position:     {pos(state.get_hand_pose(0, 0))}
        - Keys held:    {btn(left._buttons_held)}
        - Keys down:    {btn(left._buttons_down)}
        - Keys up:      {btn(left._buttons_up)}
        - Keys touched: {btn(left._buttons_touched)}
        - Hand trigger: {left.get_hand_trigger()}
        - Idx trigger:  {left.get_index_trigger()}
        - Thumbstick:   {left.get_thumbstick()}
        - In hand:      {left.get_is_controller_in_hand()}
        """
        self._app_service.text_drawer.add_text(
            text,
            TextOnScreenAlignment.TOP_LEFT,
        )

        text = f"""
        Right Controller:
        - Position:     {pos(state.get_hand_pose(0, 1))}
        - Keys held:    {btn(right._buttons_held)}
        - Keys down:    {btn(right._buttons_down)}
        - Keys up:      {btn(right._buttons_up)}
        - Keys touched: {btn(right._buttons_touched)}
        - Hand trigger: {right.get_hand_trigger()}
        - Idx trigger:  {right.get_index_trigger()}
        - Thumbstick:   {right.get_thumbstick()}
        - In hand:      {right.get_is_controller_in_hand()}
        """
        self._app_service.text_drawer.add_text(
            text,
            TextOnScreenAlignment.TOP_CENTER,
        )

        text = f"""
        HMD:
        - Position:     {pos(state.get_head_pose(0))}
        """
        self._app_service.text_drawer.add_text(
            text,
            TextOnScreenAlignment.TOP_RIGHT,
        )


@hydra.main(version_base=None, config_path="config", config_name="xr_reader")
def main(config):
    hitl_main(
        config,
        lambda app_service: AppStateXRReader(app_service),
    )


if __name__ == "__main__":
    register_hydra_plugins()
    main()
