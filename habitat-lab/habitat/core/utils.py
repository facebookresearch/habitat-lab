#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import cmath
import dataclasses
import json
import math
from typing import Any, Dict, List, Optional

import attr
import numpy as np
import quaternion  # noqa: F401
from omegaconf import OmegaConf

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
    self: Any, attribute: attr.Attribute, value: Optional[Any]
) -> None:
    if value is None:
        raise ValueError(f"Argument '{attribute.name}' must be set")


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


class DatasetJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, quaternion.quaternion):
            return quaternion_to_list(obj)
        if OmegaConf.is_config(obj):
            return OmegaConf.to_container(obj)
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)

        return (
            obj.__getstate__()
            if hasattr(obj, "__getstate__")
            else obj.__dict__
        )


class DatasetFloatJSONEncoder(DatasetJSONEncoder):
    r"""JSON Encoder that sets a float precision for a space saving purpose and
    encodes ndarray and quaternion. The encoder is compatible with JSON
    version 2.0.9.
    """

    # Overriding method to inject own `_repr` function for floats with needed
    # precision.
    def iterencode(self, o, _one_shot=False):
        markers: Optional[Dict] = {} if self.check_circular else None
        if self.ensure_ascii:
            _encoder = encode_basestring_ascii
        else:
            _encoder = encode_basestring

        default_repr = lambda x: format(x, ".5f")

        def floatstr(
            o,
            allow_nan=self.allow_nan,
            _repr=default_repr,
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
