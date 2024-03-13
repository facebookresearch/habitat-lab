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

    # pad images to avoid error in Magnum ImageView code.
    def _pad_image_tensor_for_draw_requirements(self, tensor):
        # todo: investigate Magnum ImageView2D to understand actual requirements here.
        required_width_multiple = 4

        # Calculate padding for the second-to-last dimension
        remainder = tensor.shape[-2] % required_width_multiple
        if remainder != 0:
            padding_needed = required_width_multiple - remainder
        else:
            padding_needed = 0

        # Apply padding if needed
        if padding_needed > 0:
            if isinstance(tensor, np.ndarray):
                pad_width = [
                    (0, 0)
                    if dim != len(tensor.shape) - 2
                    else (0, padding_needed)
                    for dim in range(len(tensor.shape))
                ]
                tensor = np.pad(
                    tensor,
                    pad_width=pad_width,
                    mode="constant",
                    constant_values=0,
                )
            else:
                raise AssertionError()  # Code below is untested! Uncomment it and try it!
                # import torch.nn.functional as F  # lazy import; avoid torch dependency at file scope
                # assert len(tensor.shape) == 3, "Tensor is not 3D."
                # # Only pad the second-to-last dimension (i.e., middle dimension in a 3D tensor)
                # pad = (
                #     0,
                #     0,
                #     0,
                #     0,
                #     padding_needed,
                #     0,
                # )  # format: (front, back, left, right, top, bottom)
                # tensor = F.pad(tensor, pad, "constant", 0)

        return tensor

    def draw(self, image_tensor, dest_x, dest_y):
        import torch  # lazy import; avoid torch dependency at file scope

        if isinstance(image_tensor, (np.ndarray, torch.Tensor)):
            assert len(image_tensor.shape) == 3 and (
                image_tensor.shape[2] == 3 or image_tensor.shape[2] == 4
            )
            assert (
                image_tensor.dtype == np.uint8
                or image_tensor.dtype == torch.uint8
            )

            padded_tensor = self._pad_image_tensor_for_draw_requirements(
                image_tensor
            )

            # todo: catch case where storage is not 0-dim-major?
            self.draw_bytearray(
                bytearray_pixel_data=bytearray(padded_tensor),
                height=padded_tensor.shape[0],
                width=padded_tensor.shape[1],
                bytes_per_pixel=padded_tensor.shape[2],
                dest_x=dest_x,
                dest_y=dest_y,
            )
        else:
            raise TypeError(
                "Type "
                + type(image_tensor)
                + " isn't yet supported by ImageFramebufferDrawer."
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
