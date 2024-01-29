#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class DebugVideoWriter:
    """
    Helper class to write debug images to video files.

    This is a thin wrapper over habitat_sim.utils.viz_utils.make_video. The expected
    input is exactly HitlDriver's post_sim_update_dict["debug_images"].
    """

    def __init__(self, fps=10):
        self._cached_images = None
        self._fps = fps

    def add_frame(self, debug_images_for_frame):
        if self._cached_images is None:
            self._cached_images = []
            for image in debug_images_for_frame:
                self._cached_images.append([image])
        else:
            for image_idx, image in enumerate(debug_images_for_frame):
                assert self._cached_images[image_idx][0].shape == image.shape
                self._cached_images[image_idx].append(image)

    def write(self, filepath_base):
        if self._cached_images is not None:
            import numpy as np

            from habitat_sim.utils import viz_utils as vut

            for image_idx, images in enumerate(self._cached_images):
                np_images = np.array(images)
                np_images = np_images[:, ::-1, :, :]  # flip vertically
                np_images = np.expand_dims(
                    np_images, 1
                )  # add dummy dimension required by vut.make_video
                vut.make_video(
                    np_images,
                    0,
                    "color",
                    f"{filepath_base}{image_idx}",
                    fps=self._fps,
                    open_vid=False,
                )
