#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
import numpy as np

import habitat_sim
from habitat.gui.gui_application import GuiAppRenderer
from habitat.gui.image_framebuffer_drawer import ImageFramebufferDrawer
from habitat_sim import ReplayRenderer, ReplayRendererConfiguration


class ReplayGuiAppRenderer(GuiAppRenderer):
    def __init__(
        self,
        window_width,
        window_height,
        use_batch_renderer=False,
        viewport_rect=None,
    ):
        self.window_size = mn.Vector2i(window_width, window_height)
        # arbitrary uuid
        self._sensor_uuid = "rgb_camera"

        cfg = ReplayRendererConfiguration()
        cfg.num_environments = 1
        cfg.standalone = False  # Context is owned by the GLFW window
        camera_sensor_spec = habitat_sim.CameraSensorSpec()
        camera_sensor_spec.senπsor_type = habitat_sim.SensorType.COLOR
        camera_sensor_spec.uuid = self._sensor_uuid
        if viewport_rect:
            # unfortunately, at present, we only support a viewport rect placed
            # in the bottom left corner. See https://cvmlp.slack.com/archives/G0131KVLBLL/p1682023823697029
            assert viewport_rect.left == 0
            assert viewport_rect.bottom == 0
            assert viewport_rect.right <= window_width
            assert viewport_rect.top <= window_height
            camera_sensor_spec.resolution = [
                viewport_rect.top,
                viewport_rect.right,
            ]
        else:
            camera_sensor_spec.resolution = [
                window_height,
                window_width,
            ]
        camera_sensor_spec.position = np.array([0, 0, 0])
        camera_sensor_spec.orientation = np.array([0, 0, 0])

        cfg.sensor_specifications = [camera_sensor_spec]
        cfg.gpu_device_id = 0  # todo
        cfg.force_separate_semantic_scene_graph = False
        cfg.leave_context_with_background_renderer = False
        self._replay_renderer = (
            ReplayRenderer.create_batch_replay_renderer(cfg)
            if use_batch_renderer
            else ReplayRenderer.create_classic_replay_renderer(cfg)
        )

        # todo: allocate drawer lazily
        self._image_drawer = ImageFramebufferDrawer(
            max_width=1440, max_height=1440
        )
        self._debug_images = []
        self._need_render = True

    def post_sim_update(self, post_sim_update_dict):
        keyframes = post_sim_update_dict["keyframes"]
        self.cam_transform = post_sim_update_dict["cam_transform"]

        env_index = 0
        for keyframe in keyframes:
            self._replay_renderer.set_environment_keyframe(env_index, keyframe)

        if "debug_images" in post_sim_update_dict:
            self._debug_images = post_sim_update_dict["debug_images"]

        if len(keyframes):
            self._need_render = True

    def unproject(self, viewport_pos):
        return self._replay_renderer.unproject(0, viewport_pos)

    def render_update(self, dt):
        if not self._need_render:
            return False

        transform = self.cam_transform
        env_index = 0
        self._replay_renderer.set_sensor_transform(
            env_index, self._sensor_uuid, transform
        )

        mn.gl.default_framebuffer.clear(
            mn.gl.FramebufferClear.COLOR | mn.gl.FramebufferClear.DEPTH
        )
        mn.gl.default_framebuffer.bind()

        self._replay_renderer.render(mn.gl.default_framebuffer)

        # arrange debug images on right side of frame, tiled down from the top
        dest_y = self.window_size.y
        for image in self._debug_images:
            im_height, im_width, _ = image.shape
            self._image_drawer.draw(
                image, self.window_size.x - im_width, dest_y - im_height
            )
            dest_y -= im_height

        self._need_render = False

        return True
