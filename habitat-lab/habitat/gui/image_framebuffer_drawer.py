#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
import numpy as np


class ImageFramebufferDrawer:
    def __init__(self, max_width=1024, max_height=1024):
        self._max_width = max_width
        self._max_height = max_height

        size = mn.Vector2i(self._max_width, self._max_height)
        # pre-allocate texture and framebuffer
        self.texture = mn.gl.Texture2D()
        self.texture.set_storage(1, mn.gl.TextureFormat.RGBA8, size)
        # color.set_sub_image(0, (0, 0), image)
        self.framebuffer = mn.gl.Framebuffer(mn.Range2Di((0, 0), size))
        self.framebuffer.attach_texture(
            mn.gl.Framebuffer.ColorAttachment(0), self.texture, 0
        )

    def draw(self, pixel_data, dest_x, dest_y):
        if isinstance(pixel_data, np.ndarray):
            assert (
                pixel_data.dtype == np.uint8
                and len(pixel_data.shape) == 3
                and (pixel_data.shape[2] == 3 or pixel_data.shape[2] == 4)
            )

            pixel_data_h, pixel_data_w, pixel_data_c = pixel_data.shape
            # todo: catch case where storage is not 0-dim-major?
            self.draw_bytearray(
                bytearray_pixel_data=bytearray(pixel_data),
                height=pixel_data_h,
                width=pixel_data_w,
                bytes_per_pixel=pixel_data_c,
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
        assert height <= self._max_height and width <= self._max_width, (
            f"Pixel data height={height}, width={width}, "
            f"but ImageFramebufferDrawer max_height={self._max_height} "
            f"and max_width={self._max_width}."
        )
        assert len(bytearray_pixel_data) == width * height * bytes_per_pixel
        assert bytes_per_pixel == 3 or bytes_per_pixel == 4

        # mn.ImageView2D expects four-byte-aligned rows by default
        # we pass a pixel storage https://doc.magnum.graphics/python/magnum/PixelStorage/
        # argument to alow drawing images of variable resolution
        storage = mn.PixelStorage()
        storage.alignment = 1

        size = mn.Vector2i(width, height)
        image = mn.ImageView2D(
            storage,
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
