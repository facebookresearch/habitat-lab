#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
import numpy as np

import habitat_sim
from habitat.gui.gui_application import GuiAppRenderer
from habitat.gui.image_framebuffer_drawer import ImageFramebufferDrawer
from habitat.gui.text_drawer import TextDrawer, TextOnScreenAlignment
from habitat_sim import ReplayRenderer, ReplayRendererConfiguration


class ReplayGuiAppRenderer(GuiAppRenderer):
    def __init__(
        self,
        window_size,
        viewport_rect=None,
        use_batch_renderer=False,
        im_framebuffer_drawer_kwargs=None,
        text_drawer_kwargs=None,
    ):
        self.window_size = window_size
        # arbitrary uuid
        self._sensor_uuid = "rgb_camera"

        cfg = ReplayRendererConfiguration()
        cfg.num_environments = 1
        cfg.standalone = False  # Context is owned by the GLFW window
        camera_sensor_spec = habitat_sim.CameraSensorSpec()
        camera_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        camera_sensor_spec.uuid = self._sensor_uuid
        if viewport_rect:
            # unfortunately, at present, we only support a viewport rect placed
            # in the bottom left corner. See https://cvmlp.slack.com/archives/G0131KVLBLL/p1682023823697029
            assert viewport_rect.left == 0
            assert viewport_rect.bottom == 0
            assert viewport_rect.right <= self.window_size.x
            assert viewport_rect.top <= self.window_size.y
            camera_sensor_spec.resolution = [
                viewport_rect.top,
                viewport_rect.right,
            ]
        else:
            camera_sensor_spec.resolution = [
                self.window_size.y,
                self.window_size.x,
            ]
        camera_sensor_spec.position = np.array([0, 0, 0])
        camera_sensor_spec.orientation = np.array([0, 0, 0])
        camera_sensor_spec.near = 0.2
        camera_sensor_spec.far = 50.0

        cfg.sensor_specifications = [camera_sensor_spec]
        cfg.gpu_device_id = 0  # todo
        cfg.force_separate_semantic_scene_graph = False
        cfg.leave_context_with_background_renderer = False
        # TODO:  Reenable frustum culling when replay renderer issues are solved.
        if hasattr(cfg, "enable_frustum_culling"):
            cfg.enable_frustum_culling = False
        if hasattr(cfg, "enable_hbao"):
            cfg.enable_hbao = True
        self._replay_renderer = (
            ReplayRenderer.create_batch_replay_renderer(cfg)
            if use_batch_renderer
            else ReplayRenderer.create_classic_replay_renderer(cfg)
        )

        self._debug_images = None
        self._need_render = True

        im_framebuffer_drawer_kwargs = im_framebuffer_drawer_kwargs or {}
        self._image_drawer: ImageFramebufferDrawer = ImageFramebufferDrawer(
            **im_framebuffer_drawer_kwargs
        )
        text_drawer_kwargs = text_drawer_kwargs or {}
        self._text_drawer: TextDrawer = TextDrawer(
            self.window_size, **text_drawer_kwargs
        )

    def set_image_drawer(self, image_drawer: ImageFramebufferDrawer):
        self._image_drawer = image_drawer

    def set_text_drawer(self, text_drawer: TextDrawer):
        self._text_drawer = text_drawer

    def post_sim_update(self, post_sim_update_dict):
        self._need_render = True
        keyframes = post_sim_update_dict["keyframes"]
        self.cam_transform = post_sim_update_dict["cam_transform"]

        env_index = 0
        for keyframe in keyframes:
            self._replay_renderer.set_environment_keyframe(env_index, keyframe)

        self._debug_images = (
            post_sim_update_dict["debug_images"]
            if "debug_images" in post_sim_update_dict
            else None
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

        if self._debug_images:

            def upscale_image(image, scale):
                import torch  # lazy import

                assert isinstance(scale, int) and scale > 1
                if isinstance(image, np.ndarray):
                    return np.repeat(
                        np.repeat(image, scale, axis=0), scale, axis=1
                    )
                # Check if the image is a PyTorch tensor
                elif isinstance(image, torch.Tensor):
                    return torch.repeat_interleave(
                        torch.repeat_interleave(image, scale, dim=0),
                        scale,
                        dim=1,
                    )
                # If the image is neither a numpy array nor a PyTorch tensor, raise an assertion error
                else:
                    raise AssertionError(
                        "Input image should be either a numpy array or a PyTorch tensor"
                    )

            # apply scales
            scaled_titled_images = []
            for i in range(len(self._debug_images)):
                title, src_image, scale = self._debug_images[i]
                if scale is not None and scale != 1:
                    tup = (title, upscale_image(src_image, scale))
                else:
                    tup = (title, src_image)
                scaled_titled_images.append(tup)

            max_im_width = max(
                scaled_titled_images, key=lambda tup: tup[1].shape[1]
            )[1].shape[1]

            # arrange debug images on right side of frame, tiled down from the top
            dest_y = self.window_size.y
            for title, image in scaled_titled_images:
                im_height, im_width, _ = image.shape

                # add_text y convention is: top = 0, bottom = -self.window_size.y
                text_pos_y = -(self.window_size.y - dest_y)
                text_pos_x = self.window_size.x - max_im_width
                self._text_drawer.add_text(
                    title,
                    TextOnScreenAlignment.TOP_LEFT,
                    text_pos_x,
                    text_pos_y,
                )

                text_pad_y = 40
                screen_x = self.window_size.x - im_width
                screen_y = dest_y - im_height - text_pad_y
                self._image_drawer.draw(image, screen_x, screen_y)

                dest_y -= im_height + text_pad_y

        # draws text collected in self._text_drawer._text_transform_pairs on the screen
        mn.gl.default_framebuffer.bind()
        self._text_drawer.draw_text()

        self._need_render = False

        return True
