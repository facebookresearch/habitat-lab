#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from habitat_hitl.app_states.app_state_abc import AppState
from habitat_hitl.core.key_mapping import KeyCode
from habitat_hitl.core.text_drawer import TextOnScreenAlignment
from habitat_hitl.environment.hitl_tutorial import Tutorial, generate_tutorial


class AppStateTutorial(AppState):
    def __init__(
        self,
        app_service,
    ):
        self._app_service = app_service
        self._gui_agent_ctrl = (
            self._app_service.gui_agent_controllers[0]
            if len(self._app_service.gui_agent_controllers)
            else None
        )
        self._cam_transform = None

    def get_sim(self):
        return self._app_service.sim

    def get_gui_controlled_agent_index(self):
        return self._gui_agent_ctrl._agent_idx

    def on_enter(self, final_eye_pos, final_lookat_pos):
        self._tutorial: Tutorial = generate_tutorial(
            sim=self.get_sim(),
            agent_idx=self.get_gui_controlled_agent_index(),
            final_lookat=(
                final_eye_pos,
                final_lookat_pos,
            ),
        )

        assert not self._tutorial.is_completed()
        self._cam_transform = self._tutorial.get_look_at_matrix()

    def on_environment_reset(self, episode_recorder_dict):
        pass

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

        if self._app_service.gui_input.get_key_down(KeyCode.SPACE):
            self._tutorial.skip_stage()

        if self._app_service.gui_input.get_key_down(KeyCode.Q):
            while not self._tutorial.is_completed():
                self._tutorial.skip_stage()

        if self._tutorial.is_completed():
            self._tutorial.stop_animations()
        else:
            self._cam_transform = self._tutorial.get_look_at_matrix()

    def _update_help_text(self):
        controls_str = self._tutorial.get_help_text()
        if len(controls_str) > 0:
            self._app_service.text_drawer.add_text(
                controls_str, TextOnScreenAlignment.TOP_LEFT
            )

        tutorial_str = self._tutorial.get_display_text()
        if len(tutorial_str) > 0:
            self._app_service.text_drawer.add_text(
                tutorial_str,
                TextOnScreenAlignment.TOP_CENTER,
                text_delta_x=-280,
                text_delta_y=-50,
            )

    def record_state(self):
        # Because the environment is not stepped in the tutorial, we don't need to save state.
        pass
