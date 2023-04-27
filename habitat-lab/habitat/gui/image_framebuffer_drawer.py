#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
import numpy as np


class ImageFramebufferDrawer:
    def __init__(self, max_width=1440, max_height=1440):
        size = mn.Vector2i(max_width, max_height)

        # pre-allocate texture and framebuffer
        self.texture = mn.gl.Texture2D()
        self.texture.set_storage(1, mn.gl.TextureFormat.RGBA8, size)
        # color.set_sub_image(0, (0, 0), image)
        self.framebuffer = mn.gl.Framebuffer(mn.Range2Di((0, 0), size))
        self.framebuffer.attach_texture(
            mn.gl.Framebuffer.ColorAttachment(0), self.texture, 0
        )

    def draw(self, pixel_data, dest_x, dest_y):
        import torch  # lazy import; avoid torch dependency at file scope

        if isinstance(pixel_data, (np.ndarray, torch.Tensor)):
            assert len(pixel_data.shape) == 3 and (
                pixel_data.shape[2] == 3 or pixel_data.shape[2] == 4
            )
            assert (
                pixel_data.dtype == np.uint8 or pixel_data.dtype == torch.uint8
            )
            # todo: catch case where storage is not 0-dim-major?
            self.draw_bytearray(
                bytearray_pixel_data=bytearray(pixel_data),
                height=pixel_data.shape[0],
                width=pixel_data.shape[1],
                bytes_per_pixel=pixel_data.shape[2],
                dest_x=dest_x,
                dest_y=dest_y,
            )
        else:
            raise TypeError(
                "Type "
                + type(pixel_data)
                + " isn't yet supported by ImageFramebufferDrawer. You should add it!"
            )

    def draw_bytearray(
        self,
        bytearray_pixel_data,
        height,
        width,
        bytes_per_pixel,
        dest_x,
        dest_y,
    ):
        # see max_width, max_height in constructor
        assert width <= self.texture.image_size(0)[0]
        assert height <= self.texture.image_size(0)[1]

        assert len(bytearray_pixel_data) == width * height * bytes_per_pixel
        assert bytes_per_pixel == 3 or bytes_per_pixel == 4

        size = mn.Vector2i(width, height)
        image = mn.ImageView2D(
            mn.PixelFormat.RGBA8_UNORM
            if bytes_per_pixel == 4
            else mn.PixelFormat.RGB8_UNORM,
            size,
            bytearray_pixel_data,
        )
        self.texture.set_sub_image(0, (0, 0), image)

        dest_coord = mn.Vector2i(dest_x, dest_y)
        mn.gl.AbstractFramebuffer.blit(
            self.framebuffer,
            mn.gl.default_framebuffer,
            mn.Range2Di((0, 0), size),
            mn.Range2Di(dest_coord, dest_coord + size),
            mn.gl.FramebufferBlit.COLOR,
            mn.gl.FramebufferBlitFilter.NEAREST,
        )
