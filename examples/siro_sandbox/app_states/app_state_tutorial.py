#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
from app_states.app_state_abc import AppState
from camera_helper import CameraHelper
from hablab_utils import get_agent_art_obj_transform
from hitl_tutorial import Tutorial, generate_tutorial

from habitat.gui.gui_input import GuiInput
from habitat.gui.text_drawer import TextOnScreenAlignment


class AppStateTutorial(AppState):
    def __init__(
        self,
        sandbox_service,
        gui_agent_ctrl,
    ):
        self._sandbox_service = sandbox_service
        self._gui_agent_ctrl = gui_agent_ctrl
        self._cam_transform = None
        self._camera_helper = CameraHelper(
            self._sandbox_service.args, self._sandbox_service.gui_input
        )

    def get_sim(self):
        return self._sandbox_service.sim

    def get_gui_controlled_agent_index(self):
        return self._gui_agent_ctrl._agent_idx

    def _get_camera_lookat_pos(self):
        agent_root = get_agent_art_obj_transform(
            self.get_sim(), self.get_gui_controlled_agent_index()
        )
        lookat = agent_root.translation + mn.Vector3(0, 1, 0)
        return lookat

    def on_environment_reset(self, episode_recorder_dict):
        base_pos = self._get_camera_lookat_pos()
        self._camera_helper.update(base_pos, None)
        self._cam_transform = self._camera_helper.get_cam_transform()

        (eye_pos, lookat_pos) = self._camera_helper._get_eye_and_lookat(
            base_pos
        )
        self._tutorial: Tutorial = generate_tutorial(
            sim=self.get_sim(),
            agent_idx=self.get_gui_controlled_agent_index(),
            final_lookat=(eye_pos, lookat_pos),
        )

    def sim_update(self, dt, post_sim_update_dict):
        self._sim_update_tutorial(dt)

        post_sim_update_dict["cam_transform"] = self._cam_transform

        if not self._tutorial.is_completed():
            self._update_help_text()

    def _sim_update_tutorial(self, dt: float):
        # todo: get rid of this
        # Keyframes are saved by RearrangeSim when stepping the environment.
        # Because the environment is not stepped in the tutorial, we need to save keyframes manually for replay rendering to work.
        self.get_sim().gfx_replay_manager.save_keyframe()

        self._tutorial.update(dt)

        if self._sandbox_service.gui_input.get_key_down(GuiInput.KeyNS.SPACE):
            self._tutorial.skip_stage()

        if self._tutorial.is_completed():
            self._tutorial.stop_animations()
        else:
            self._cam_transform = self._tutorial.get_look_at_matrix()

    def _update_help_text(self):
        controls_str = self._tutorial.get_help_text()
        if len(controls_str) > 0:
            self._sandbox_service.text_drawer.add_text(
                controls_str, TextOnScreenAlignment.TOP_LEFT
            )

        tutorial_str = self._tutorial.get_display_text()
        if len(tutorial_str) > 0:
            self._sandbox_service.text_drawer.add_text(
                tutorial_str,
                TextOnScreenAlignment.TOP_CENTER,
                text_delta_x=-280,
                text_delta_y=-50,
            )

    def record_state(self):
        # Because the environment is not stepped in the tutorial, we don't need to save state.
        pass

    def is_app_state_done(self):
        """Returns True if all the tutorial stages are completed."""
        return self._tutorial.is_completed()
