#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import numpy as np


def tile_images(images: List[np.ndarray]) -> np.ndarray:
    r"""Tile multiple images into single image

    Args:
        images: list of images where each image has dimension
            (height x width x channels)

    Returns:
        tiled image (new_height x width x channels)
    """
    assert len(images) > 0, "empty list of images"
    np_images = np.asarray(images)
    n_images, height, width, n_channels = np_images.shape
    new_height = int(np.ceil(np.sqrt(n_images)))
    new_width = int(np.ceil(float(n_images) / new_height))
    # pad with empty images to complete the rectangle
    np_images = np.array(
        images
        + [images[0] * 0 for _ in range(n_images, new_height * new_width)]
    )
    # img_HWhwc
    out_image = np_images.reshape(
        new_height, new_width, height, width, n_channels
    )
    # img_HhWwc
    out_image = out_image.transpose(0, 2, 1, 3, 4)
    # img_Hh_Ww_c
    out_image = out_image.reshape(
        new_height * height, new_width * width, n_channels
    )
    return out_image


def not_none_validator(self, attribute, value):
    if value is None:
        raise ValueError(f"Argument '{attribute.name}' must be set")


def try_cv2_import():
    r"""The PyRobot python3 version which is a dependency of Habitat-PyRobot integration
    relies on ROS running in python2.7. In order to import cv2 in python3 we need to remove
    the python2.7 path from sys.path. To use the Habitat-PyRobot integration the user
    needs to export environment variable ROS_PATH which will look something like:
    /opt/ros/kinetic/lib/python2.7/dist-packages
    """
    import sys
    import os

    ros_path = os.environ.get("ROS_PATH")
    if ros_path is not None and ros_path in sys.path:
        sys.path.remove(ros_path)
        import cv2
        sys.path.append(ros_path)
    else:
        import cv2

    return cv2


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs
            )
        return cls._instances[cls]
