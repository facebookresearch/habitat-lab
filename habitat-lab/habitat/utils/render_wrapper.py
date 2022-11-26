#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import cv2
import numpy as np

from habitat.utils.gym_adapter import flatten_dict


def append_text_to_image(
    image: np.ndarray, text: List[str], font_size: float = 0.5
):
    r"""Appends lines of text on top of an image. First this will render to the
    left-hand side of the image, once that column is full, it will render to
    the right hand-side of the image.
    :param image: the image to put text underneath
    :param text: The list of strings which will be rendered, separated by new lines.
    :returns: A new image with text inserted underneath the input image
    """
    h, w, c = image.shape
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX

    y = 0
    left_aligned = True
    for line in text:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        if y > h:
            left_aligned = False
            y = textsize[1] + 10

        if left_aligned:
            x = 10
        else:
            x = w - (textsize[0] + 10)

        cv2.putText(
            image,
            line,
            (x, y),
            font,
            font_size,
            (0, 0, 0),
            font_thickness * 2,
            lineType=cv2.LINE_AA,
        )

        cv2.putText(
            image,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    return np.clip(image, 0, 255)


def overlay_frame(frame, info, additional=None):
    lines = []
    flattened_info = flatten_dict(info)
    for k, v in flattened_info.items():
        if isinstance(v, str):
            lines.append(f"{k}: {v}")
        else:
            lines.append(f"{k}: {v:.2f}")
    if additional is not None:
        lines.extend(additional)

    frame = append_text_to_image(frame, lines, font_size=0.25)

    return frame
