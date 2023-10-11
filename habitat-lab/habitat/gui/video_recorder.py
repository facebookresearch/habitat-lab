#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

import magnum as mn
import numpy as np

import habitat_sim

DEFAULT_OUTPUT_FILE_PATH: str = "video"
OBSERVATION_PLACEHOLDER_NAME: str = "s"  # Dummy sensor name
OBSERVATION_TYPE: str = "color"
FPS: int = 30


class VideoRecorder:
    _output_file_path: str

    _recording_images: List[Dict[str, np.ndarray]] = []
    _gpu_to_cpu_image: mn.MutableImageView2D = None
    _gpu_to_cpu_buffer: np.ndarray = None
    _recording_video: bool = False

    def __init__(self, output_file_path: str = DEFAULT_OUTPUT_FILE_PATH):
        self._output_file_path = output_file_path

    # Start recording the video.
    def start_video_recording(self):
        self._recording_video = True

    # Save video and exit.
    def save_video(self):
        if self._recording_video and len(self._recording_images) > 0:
            habitat_sim.utils.viz_utils.make_video(
                observations=self._recording_images,
                primary_obs=OBSERVATION_PLACEHOLDER_NAME,
                primary_obs_type=OBSERVATION_TYPE,
                video_file=self._output_file_path,
                fps=FPS,
            )
            exit()
        else:
            print("No frame recorded. Press '-' to start recording video.")

    # Record a frame.
    def record_video_frame(self):
        if self._recording_video:
            viewport = mn.gl.default_framebuffer.viewport

            # Lazy allocation of video recording buffer and image view.
            if self._gpu_to_cpu_buffer is None:
                self._gpu_to_cpu_buffer = np.empty(
                    (
                        viewport.size_y(),
                        viewport.size_x(),
                        3,
                    ),
                    dtype=np.uint8,
                )
                self._gpu_to_cpu_image = mn.MutableImageView2D(
                    mn.PixelFormat.RGB8_UNORM,
                    [
                        viewport.size_x(),
                        viewport.size_y(),
                    ],
                    self._gpu_to_cpu_buffer,
                )
                # Flip the view vertically for presentation
                self._gpu_to_cpu_buffer = np.flip(
                    self._gpu_to_cpu_buffer.view(), axis=0
                )

            # Record frame.
            rect = mn.Range2Di(
                mn.Vector2i(),
                mn.Vector2i(viewport.size_x(), viewport.size_y()),
            )
            mn.gl.default_framebuffer.read(rect, self._gpu_to_cpu_image)
            self._recording_images.append(
                {OBSERVATION_PLACEHOLDER_NAME: self._gpu_to_cpu_buffer.copy()}
            )
