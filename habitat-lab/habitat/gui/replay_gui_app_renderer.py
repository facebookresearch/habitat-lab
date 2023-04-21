#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from pprint import pformat

import magnum as mn
import numpy as np

import habitat_sim
from habitat.gui.gui_application import GuiAppRenderer
from habitat.gui.image_framebuffer_drawer import ImageFramebufferDrawer
from habitat.gui.text_drawer import TextDrawer
from habitat_sim import ReplayRenderer, ReplayRendererConfiguration


class ReplayGuiAppRenderer(GuiAppRenderer):
    def __init__(
        self,
        viewport_size,
        use_batch_renderer=False,
        im_framebuffer_drawer_kwargs=None,
        text_drawer_kwargs=None,
    ):
        self.viewport_size = viewport_size
        # arbitrary uuid
        self._sensor_uuid = "rgb_camera"

        cfg = ReplayRendererConfiguration()
        cfg.num_environments = 1
        cfg.standalone = False  # Context is owned by the GLFW window
        camera_sensor_spec = habitat_sim.CameraSensorSpec()
        camera_sensor_spec.senπsor_type = habitat_sim.SensorType.COLOR
        camera_sensor_spec.uuid = self._sensor_uuid
        camera_sensor_spec.resolution = [
            self.viewport_size.y,
            self.viewport_size.x,
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

        self._debug_images = []
        self._need_render = True

        im_framebuffer_drawer_kwargs = im_framebuffer_drawer_kwargs or {}
        self._image_drawer: ImageFramebufferDrawer = ImageFramebufferDrawer(
            **im_framebuffer_drawer_kwargs
        )
        text_drawer_kwargs = text_drawer_kwargs or {}
        self._text_drawer: TextDrawer = TextDrawer(
            viewport_size, **text_drawer_kwargs
        )

    def set_image_drawer(self, image_drawer: ImageFramebufferDrawer):
        self._image_drawer = image_drawer

    def set_text_drawer(self, text_drawer: TextDrawer):
        self._text_drawer = text_drawer

    def post_sim_update(self, post_sim_update_dict):
        keyframes = post_sim_update_dict["keyframes"]
        self.cam_transform = post_sim_update_dict["cam_transform"]

        env_index = 0
        for keyframe in keyframes:
            self._replay_renderer.set_environment_keyframe(env_index, keyframe)

        if len(keyframes):
            self._need_render = True

        if "debug_images" in post_sim_update_dict:
            self._debug_images = post_sim_update_dict["debug_images"]

        if "info" in post_sim_update_dict:
            # get needed metrics and format text string
            # to be drawn on the GUI app screen
            info = post_sim_update_dict["info"]
            self._text_to_draw = (
                pformat(
                    {
                        "App metrics": info["app"],
                        "Env metrics": info["env"],
                    },
                )
                .replace("'", " ")
                .replace("{", " ")
                .replace("}", " ")
            )

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

        # draws text collected in self._text_drawer._text_transform_pairs on the screen
        self._text_drawer.draw_text()

        # arrange debug images on right side of frame, tiled down from the top
        dest_y = self.viewport_size.y
        for image in self._debug_images:
            im_height, im_width, _ = image.shape
            self._image_drawer.draw(
                image, self.viewport_size.x - im_width, dest_y - im_height
            )
            dest_y -= im_height

        self._need_render = False

        return True
