#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import cmath
import json
import math
from typing import Any, Dict, List, Optional

import numpy as np
import quaternion  # noqa: F401

from habitat.utils.geometry_utils import quaternion_to_list

# Internals from inner json library needed for patching functionality in
# DatasetFloatJSONEncoder.
try:
    from _json import encode_basestring_ascii  # type: ignore
except ImportError:
    encode_basestring_ascii = None  # type: ignore
try:
    from _json import encode_basestring  # type: ignore
except ImportError:
    encode_basestring = None  # type: ignore


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


def not_none_validator(
    self: Any, attribute: Any, value: Optional[Any]
) -> None:
    if value is None:
        raise ValueError(f"Argument '{attribute.name}' must be set")


def try_cv2_import():
    r"""The PyRobot python3 version which is a dependency of Habitat-PyRobot integration
    relies on ROS running in python2.7. In order to import cv2 in python3 we need to remove
    the python2.7 path from sys.path. To use the Habitat-PyRobot integration the user
    needs to export environment variable ROS_PATH which will look something like:
    /opt/ros/kinetic/lib/python2.7/dist-packages
    """
    import os
    import sys

    ros_path = os.environ.get("ROS_PATH")
    if ros_path is not None and ros_path in sys.path:
        sys.path.remove(ros_path)
        import cv2

        sys.path.append(ros_path)
    else:
        import cv2

    return cv2


class Singleton(type):
    _instances: Dict["Singleton", "Singleton"] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs
            )
        return cls._instances[cls]


def center_crop(obs, new_shape):
    top_left = (
        (obs.shape[0] // 2) - (new_shape[0] // 2),
        (obs.shape[1] // 2) - (new_shape[1] // 2),
    )
    bottom_right = (
        (obs.shape[0] // 2) + (new_shape[0] // 2),
        (obs.shape[1] // 2) + (new_shape[1] // 2),
    )
    obs = obs[top_left[0] : bottom_right[0], top_left[1] : bottom_right[1], :]

    return obs


class DatasetFloatJSONEncoder(json.JSONEncoder):
    r"""JSON Encoder that sets a float precision for a space saving purpose and
    encodes ndarray and quaternion. The encoder is compatible with JSON
    version 2.0.9.
    """

    def default(self, obj):
        # JSON doesn't support numpy ndarray and quaternion
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.quaternion):
            return quaternion_to_list(obj)

        return (
            obj.__getstate__()
            if hasattr(obj, "__getstate__")
            else obj.__dict__
        )

    # Overriding method to inject own `_repr` function for floats with needed
    # precision.
    def iterencode(self, o, _one_shot=False):

        markers: Optional[Dict] = {} if self.check_circular else None
        if self.ensure_ascii:
            _encoder = encode_basestring_ascii
        else:
            _encoder = encode_basestring

        def floatstr(
            o,
            allow_nan=self.allow_nan,
            _repr=lambda x: format(x, ".5f"),
            _inf=math.inf,
            _neginf=-math.inf,
        ):
            if cmath.isnan(o):
                text = "NaN"
            elif o == _inf:
                text = "Infinity"
            elif o == _neginf:
                text = "-Infinity"
            else:
                return _repr(o)

            if not allow_nan:
                raise ValueError(
                    "Out of range float values are not JSON compliant: "
                    + repr(o)
                )

            return text

        _iterencode = json.encoder._make_iterencode(  # type: ignore
            markers,
            self.default,
            _encoder,
            self.indent,
            floatstr,
            self.key_separator,
            self.item_separator,
            self.sort_keys,
            self.skipkeys,
            _one_shot,
        )
        return _iterencode(o, 0)
