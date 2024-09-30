#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hydra
import magnum as mn
import json

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

        # "./file_scripted_replays/ep832_female_3/replay.gfx_replay.json"
        self._video_output_prefix = self._gfx_replay_viewer_cfg.replay_filepath.removeprefix(
            "./file_scripted_replays/").removesuffix("/replay.gfx_replay.json")

        self._camera_helper = CameraHelper(
            self._app_service.hitl_config, self._app_service.gui_input
        )
        # not supported for pick_throw_vr
        assert not self._app_service.hitl_config.camera.first_person_mode

        self._camera_lookat_base_pos = mn.Vector3(0, 0, 0)
        self._camera_lookat_y_offset = 1.0
        self._camera_helper.update(self._get_camera_lookat_pos(), 0.0)

        self._is_recording = False
        self._hide_gui = False

        self._camera_follow_key = "agent0"

        self._replay_user_positions = None
        if self._gfx_replay_viewer_cfg.text_overlays_path:
            with open(self._gfx_replay_viewer_cfg.text_overlays_path, 'r') as f:
                text_overlays_obj = json.load(f)
                self._text_overlays = text_overlays_obj["overlays"]

            with open(self._gfx_replay_viewer_cfg.replay_filepath, 'r') as f:
                gfx_replay_obj = json.load(f)
                gfx_replay_keyframes = gfx_replay_obj["keyframes"]

            self._num_keyframes = len(gfx_replay_keyframes)
            self._replay_user_positions = {}
            for keyframe_idx in range(self._num_keyframes):
                keyframe_user_transforms = gfx_replay_keyframes[keyframe_idx]["userTransforms"]
                for user_transform in keyframe_user_transforms:
                    key = user_transform["name"]
                    position = user_transform["transform"]["translation"]
                    if key not in self._replay_user_positions:
                        assert keyframe_idx == 0
                        self._replay_user_positions[key] = []
                    self._replay_user_positions[key].append(position)

            for key in self._replay_user_positions:
                assert len(self._replay_user_positions[key]) == self._num_keyframes


    def draw_debug_nav_lines(self):

        if not self._replay_user_positions:
            return

        agent_colors = [
            mn.Color3(1, 0.75, 0),
            mn.Color3(0, 0.5, 1.0)
        ]

        gui_drawer = self._app_service.gui_drawer

        for agent_id in [0, 1]:
            points = []
            agent_key = f"agent{agent_id}"
            agent_side = "left" if agent_id == 0 else "right"
            agent_positions = self._replay_user_positions[agent_key]

            end_keyframe = self._num_keyframes
            for overlay in self._text_overlays:
                if overlay["frame"] > self._keyframe_index and overlay["side"] == agent_side:
                    end_keyframe = overlay["frame"] + 1
                    break

            for keyframe_idx in range(self._keyframe_index, end_keyframe):
                pos = agent_positions[keyframe_idx]
                points.append(pos)

            radius = 0.5
            color = agent_colors[agent_id]
            if len(points) >= 2:
                gui_drawer.draw_path_with_endpoint_circles(points, radius, color)
            elif len(points) >= 1:
                gui_drawer.draw_circle(points[0], radius, color)


    def _get_controls_text(self):
        controls_str: str = ""
        controls_str += "T: toggle camera lookat\n"
        controls_str += "Y: play and record\n"
        controls_str += "H: hide GUI\n"
        controls_str += "7: rewind to start\n"
        controls_str += "8: jump back\n"
        controls_str += "9: hold to play\n"
        controls_str += "0: jump forward\n"
        controls_str += "ESC: exit\n"
        return controls_str

    def _get_status_text(self):
        status_str = ""
        status_str += f"keyframe: {self._keyframe_index}\n"
        status_str += f"camera: {self._camera_follow_key if self._camera_follow_key else 'free'}"
        lookat_pos = self._get_camera_lookat_pos()
        status_str += f"({lookat_pos.x:.1f}, {lookat_pos.y:.1f}, {lookat_pos.z:.1f})\n"
        return status_str

    def _update_help_text(self):
        if self._hide_gui:
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

    def _get_camera_lookat_pos(self):
        return self._camera_lookat_base_pos + mn.Vector3(0, self._camera_lookat_y_offset, 0)

    def _update_camera_lookat_base_pos(self):

        gui_input = self._app_service.gui_input
        y_speed = 0.05
        if gui_input.get_key_down(KeyCode.Z):
            self._camera_lookat_y_offset -= y_speed
        if gui_input.get_key_down(KeyCode.X):
            self._camera_lookat_y_offset += y_speed

        if self._camera_follow_key:
            tuple = self._player.get_user_transform(self._camera_follow_key)
            if tuple:
                self._camera_lookat_base_pos = tuple[0]
        else:

            xz_forward = self._camera_helper.get_xz_forward()
            xz_right = mn.Vector3(-xz_forward.z, 0.0, xz_forward.x)
            speed = self._gfx_replay_viewer_cfg.camera_move_speed * self._camera_helper.get_cam_zoom_dist()
            if gui_input.get_key(KeyCode.W):
                self._camera_lookat_base_pos += (
                    xz_forward * speed
                )
            if gui_input.get_key(KeyCode.S):
                self._camera_lookat_base_pos -= (
                    xz_forward * speed
                )
            if gui_input.get_key(KeyCode.E):
                self._camera_lookat_base_pos += (
                    xz_right * speed
                )
            if gui_input.get_key(KeyCode.Q):
                self._camera_lookat_base_pos -= (
                    xz_right * speed
                )


    def sim_update(self, dt, post_sim_update_dict):
        gui_input = self._app_service.gui_input
        if gui_input.get_key_down(KeyCode.ESC):
            post_sim_update_dict["application_exit"] = True

        # for step_size, key_dec, key_inc in [(1, KeyCode.SEVEN, KeyCode.EIGHT), (16, KeyCode.NINE, KeyCode.ZERO)]:
        #     if gui_input.get_key_down(key_dec) and self._keyframe_index > 0:
        #         self._keyframe_index = max(self._keyframe_index - step_size, 0)
        #         self._player.set_keyframe_index(self._keyframe_index)
        #     if gui_input.get_key(key_inc) and self._keyframe_index < self._player.get_num_keyframes() - 1:
        #         self._keyframe_index = min(self._keyframe_index + step_size, self._player.get_num_keyframes() - 1)
        #         self._player.set_keyframe_index(self._keyframe_index)

        if gui_input.get_key_down(KeyCode.T):
            if self._camera_follow_key == "agent0":
                self._camera_follow_key = "agent1"
            elif self._camera_follow_key == "agent1":
                self._camera_follow_key = None
            else:
                self._camera_follow_key = "agent0"

        if self._is_recording:
            self._keyframe_index += 1
            reached_end = False
            if self._keyframe_index == self._player.get_num_keyframes():
                self._keyframe_index = self._player.get_num_keyframes() - 1
                reached_end = True
            self._player.set_keyframe_index(self._keyframe_index)
            if gui_input.get_key_down(KeyCode.Y) or reached_end:
                self._app_service.video_recorder.stop_recording_and_save_video(self._video_output_prefix)
                self._is_recording = False
                self._hide_gui = False
        else:
            if gui_input.get_key_down(KeyCode.Y):
                self._app_service.video_recorder.start_recording()
                self._is_recording = True
                self._hide_gui = True
            else:
                if gui_input.get_key_down(KeyCode.H):
                    self._hide_gui = not self._hide_gui
                # seek controls only active when not recording
                if gui_input.get_key_down(KeyCode.SEVEN):
                    self._keyframe_index = 0
                elif gui_input.get_key_down(KeyCode.EIGHT):
                    self._keyframe_index = max(self._keyframe_index - 30, 0)
                elif gui_input.get_key(KeyCode.NINE):  # note use of get_key so you can hold this key
                    self._keyframe_index = min(self._keyframe_index + 1, self._player.get_num_keyframes() - 1)
                elif gui_input.get_key(KeyCode.ZERO):
                    self._keyframe_index = min(self._keyframe_index + 8, self._player.get_num_keyframes() - 1)
                self._player.set_keyframe_index(self._keyframe_index)

        self._update_camera_lookat_base_pos()

        self._camera_helper.update(self._get_camera_lookat_pos(), dt)
        self._cam_transform = self._camera_helper.get_cam_transform()
        post_sim_update_dict["cam_transform"] = self._cam_transform

        if not self._hide_gui:
            lookat_ring_radius = 0.1
            lookat_ring_color = mn.Color3(1, 0.75, 0)
            self._app_service.gui_drawer.draw_circle(
                self._camera_lookat_base_pos,
                lookat_ring_radius,
                lookat_ring_color,
            )

        self.draw_debug_nav_lines()

        self._update_help_text()


@hydra.main(
    version_base=None, config_path="./", config_name="gfx_replay_viewer"
)
def main(config):
    hitl_main(config, lambda app_service: AppStateGfxReplayViewer(app_service))


if __name__ == "__main__":
    register_hydra_plugins()
    main()
