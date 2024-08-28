#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""A module providing some disconnected core utilities."""
import cmath
import dataclasses
import json
import math
from typing import Any, Callable, Dict, List, Optional

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
    """Tile multiple images into single image

    NOTE: “candidate for deprecation”: possible duplicate function at habitat-lab/habitat/utils/visualizations/utils.py

    :param images: list of images where each image has dimension (height x width x channels)
    :return: tiled image (new_height x width x channels)
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
    """
    Attr validator to raise a ValueError if a particular attribute's value is None.

    :param self: The calling attrs class instance.
    :param attribute: The attribute to check in order to provide a precise error message.
    :param value: The value to check.

    See https://www.attrs.org/en/stable/examples.html#validators
    """
    if value is None:
        raise ValueError(f"Argument '{attribute.name}' must be set")


class Singleton(type):
    """
    This metatclass creates Singleton objects by ensuring only one instance is created and any call is directed to that instance. The mro() function and following dunders, EXCEPT __call__, are inherited from the the stdlib Python library, which defines the "type" class.
    """

    _instances: Dict["Singleton", "Singleton"] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs
            )
        return cls._instances[cls]


class DatasetJSONEncoder(json.JSONEncoder):
    """Extension of base JSONEncoder to handle common Dataset types: numpy array, numpy quaternion, Omegaconf, and dataclass."""

    def default(self, obj: Any) -> Any:
        """
        Constructs and returns a default serializable JSON object for a particular object or type obj.
        This override supports types: np.ndarray, numpy quaternion, OmegaConf DictConfig, and DataClasses.

        :param obj: The object to serialize.
        :return: The serialized JSON object.
        """
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
    """JSON Encoder that sets a float precision for a space saving purpose and
    encodes ndarray and quaternion. The encoder is compatible with JSON
    version 2.0.9.
    """

    def iterencode(self, o, _one_shot=False) -> str:
        """
        Overriding method to inject own `_repr` function for floats with needed precision.

        :param o: The float to convert.
        :param _one_shot: undocumented base JSONEncoder param. Seems to limit recursion in exchange for cmake operation.
        :return: The string rep.
        """
        markers: Optional[Dict] = {} if self.check_circular else None
        if self.ensure_ascii:
            _encoder = encode_basestring_ascii
        else:
            _encoder = encode_basestring

        default_repr = lambda x: format(x, ".5f")

        def floatstr(
            o: float,
            allow_nan: bool = self.allow_nan,
            _repr: Callable = default_repr,
            _inf: float = math.inf,
            _neginf: float = -math.inf,
        ):
            """
            Converts a float to a JSON string, handling edge cases and non-numbers.

            :param o: the float to convert
            :param allow_nan: whether or not to allow non-numeric values
            :param _repr: the function for converting numeric floats to string
            :param _inf: the value of infinite for equality check
            :param _neginf: the value of negative infinite for equality check

            :return: The string rep of the float
            """
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
