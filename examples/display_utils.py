#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import numpy as np


def display_grayscale(image):
    img_bgr = np.repeat(image, 3, 2)
    cv2.imshow("Depth Sensor", img_bgr)
    return cv2.waitKey(0)


def display_rgb(image):
    img_bgr = image[..., ::-1]
    cv2.imshow("RGB", img_bgr)
    return cv2.waitKey(0)
