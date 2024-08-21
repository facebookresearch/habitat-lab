#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import magnum as mn
import numpy as np

from habitat_hitl.core.gui_input import GuiInput
from habitat_hitl.core.key_mapping import KeyCode, MouseButton


class CameraHelper:
    def __init__(self, hitl_config, gui_input: GuiInput):
        # lookat offset yaw (spin left/right) and pitch (up/down)
        # to enable camera rotation and pitch control
        self._first_person_mode = hitl_config.camera.first_person_mode
        if self._first_person_mode:
            self._lookat_offset_yaw = 0.0
            self._lookat_offset_pitch = float(
                mn.Rad(mn.Deg(20.0))
            )  # look slightly down
            self._min_lookat_offset_pitch = (
                -max(
                    min(
                        np.radians(hitl_config.camera.max_look_up_angle),
                        np.pi / 2,
                    ),
                    0,
                )
                + 1e-5
            )
            self._max_lookat_offset_pitch = (
                -min(
                    max(
                        np.radians(hitl_config.camera.min_look_down_angle),
                        -np.pi / 2,
                    ),
                    0,
                )
                - 1e-5
            )
        else:
            # (computed from previously hardcoded mn.Vector3(0.5, 1, 0.5).normalized())
            self._lookat_offset_yaw = 0.785
            self._lookat_offset_pitch = 0.955
            self._min_lookat_offset_pitch = -np.pi / 2 + 1e-5
            self._max_lookat_offset_pitch = np.pi / 2 - 1e-5

        self.cam_zoom_dist = 1.0
        self._max_zoom_dist = 50.0
        self._min_zoom_dist = 0.02
        self._eye_pos: Optional[mn.Vector3] = None
        self._lookat_pos: Optional[mn.Vector3] = None
        self._cam_transform: Optional[mn.Matrix4] = None
        self._gui_input = gui_input

    def _camera_pitch_and_yaw_wasd_control(self):
        # update yaw and pitch using ADIK keys
        cam_rot_angle = 0.1

        if self._gui_input.get_key(KeyCode.I):
            self._lookat_offset_pitch -= cam_rot_angle
        if self._gui_input.get_key(KeyCode.K):
            self._lookat_offset_pitch += cam_rot_angle
        self._lookat_offset_pitch = np.clip(
            self._lookat_offset_pitch,
            self._min_lookat_offset_pitch,
            self._max_lookat_offset_pitch,
        )
        if self._gui_input.get_key(KeyCode.A):
            self._lookat_offset_yaw -= cam_rot_angle
        if self._gui_input.get_key(KeyCode.D):
            self._lookat_offset_yaw += cam_rot_angle

    def _camera_pitch_and_yaw_mouse_control(self):
        enable_mouse_control = self._gui_input.get_key(
            KeyCode.R
        ) or self._gui_input.get_mouse_button(MouseButton.MIDDLE)

        if enable_mouse_control:
            # update yaw and pitch by scale * mouse relative position delta
            scale = 0.003
            self._lookat_offset_yaw += (
                scale * self._gui_input.relative_mouse_position[0]
            )
            self._lookat_offset_pitch += (
                scale * self._gui_input.relative_mouse_position[1]
            )
            self._lookat_offset_pitch = np.clip(
                self._lookat_offset_pitch,
                self._min_lookat_offset_pitch,
                self._max_lookat_offset_pitch,
            )

    def _get_eye_and_lookat(self, base_pos) -> Tuple[mn.Vector3, mn.Vector3]:
        offset = mn.Vector3(
            np.cos(self.lookat_offset_yaw) * np.cos(self.lookat_offset_pitch),
            np.sin(self.lookat_offset_pitch),
            np.sin(self.lookat_offset_yaw) * np.cos(self.lookat_offset_pitch),
        )

        if self._first_person_mode:
            eye_pos = base_pos
            lookat_pos = base_pos + -offset.normalized()
        else:
            eye_pos = base_pos + offset.normalized() * self.cam_zoom_dist
            lookat_pos = base_pos
        return eye_pos, lookat_pos

    def update(self, base_pos, dt):
        if (
            not self._first_person_mode
            and self._gui_input.mouse_scroll_offset != 0
        ):
            zoom_sensitivity = 0.07
            if self._gui_input.mouse_scroll_offset < 0:
                self.cam_zoom_dist *= (
                    1.0
                    + -self._gui_input.mouse_scroll_offset * zoom_sensitivity
                )
            else:
                self.cam_zoom_dist /= (
                    1.0
                    + self._gui_input.mouse_scroll_offset * zoom_sensitivity
                )
            self.cam_zoom_dist = mn.math.clamp(
                self.cam_zoom_dist,
                self._min_zoom_dist,
                self._max_zoom_dist,
            )

        # two ways for camera pitch and yaw control for UX comparison:
        # 1) press/hold ADIK keys
        self._camera_pitch_and_yaw_wasd_control()
        # 2) press left mouse button and move mouse
        self._camera_pitch_and_yaw_mouse_control()

        self._eye_pos, self._lookat_pos = self._get_eye_and_lookat(base_pos)
        self._cam_transform = mn.Matrix4.look_at(
            self._eye_pos, self._lookat_pos, mn.Vector3(0, 1, 0)
        )

    def get_xz_forward(self):
        assert self._cam_transform
        forward_dir = self._cam_transform.transform_vector(
            -mn.Vector3(0, 0, 1)
        )
        forward_dir.y = 0
        # todo: handle case of degenerate zero vector here due to camera looking
        # straight up or down
        forward_dir = forward_dir.normalized()
        return forward_dir

    def get_cam_forward_vector(self) -> Optional[mn.Vector3]:
        assert self._cam_transform
        forward_dir = self._cam_transform.transform_vector(
            -mn.Vector3(0, 0, 1)
        )
        forward_dir = forward_dir.normalized()
        return forward_dir

    def get_cam_transform(self) -> Optional[mn.Matrix4]:
        assert self._cam_transform
        return self._cam_transform

    def get_eye_pos(self) -> Optional[mn.Vector3]:
        assert self._eye_pos
        return self._eye_pos

    def get_lookat_pos(self) -> Optional[mn.Vector3]:
        assert self._lookat_pos
        return self._lookat_pos

    @property
    def lookat_offset_yaw(self):
        return self._to_zero_2pi_range(self._lookat_offset_yaw)

    @property
    def lookat_offset_pitch(self):
        return self._lookat_offset_pitch

    @staticmethod
    def _to_zero_2pi_range(radians):
        """Helper method to properly clip radians to [0, 2pi] range."""
        return (
            (2 * np.pi) - ((-radians) % (2 * np.pi))
            if radians < 0
            else radians % (2 * np.pi)
        )
