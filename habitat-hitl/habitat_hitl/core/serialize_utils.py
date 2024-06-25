#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copyreg
import gzip
import json
import os
import pickle
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, List

import magnum as mn
import numpy as np


def unpickle_vector3(data):
    return mn.Vector3(data)


def pickle_vector3(obj):
    pickled_data = list(obj)
    return unpickle_vector3, (pickled_data,)


# fix for unpicklable type; run once at startup
copyreg.pickle(mn.Vector3, pickle_vector3)


def convert_to_json_friendly(obj):
    if isinstance(obj, (int, str, bool, type(None))):
        # If obj is a simple data type (except float), return it as is
        return obj
    elif isinstance(obj, float):
        # If obj is a simple float, round to 5 decimal places
        return round(obj, 5)
    elif isinstance(obj, (list, np.ndarray, tuple)):
        # If obj is a list or ndarray, recursively convert its elements
        return [convert_to_json_friendly(item) for item in obj]
    elif isinstance(obj, dict):
        # If obj is a dictionary, recursively convert its values
        return {
            key: convert_to_json_friendly(value) for key, value in obj.items()
        }
    elif isinstance(obj, np.generic):
        # If obj is a NumPy scalar type, convert it to a standard Python scalar type
        return convert_to_json_friendly(obj.item())
    elif isinstance(obj, datetime):
        # convert datetime to repr string
        return convert_to_json_friendly(repr(obj))
    elif isinstance(obj, mn.Vector3):
        # convert Vector3 to list
        return convert_to_json_friendly(list(obj))
    else:
        # If obj is a complex object, convert its attributes to a dictionary
        attributes = {}
        for attr in dir(obj):
            try:
                if not attr.startswith("__") and not callable(
                    getattr(obj, attr)
                ):
                    attributes[attr] = getattr(obj, attr)
            except Exception as e:
                print(
                    f"Unable to convert attribute to JSON: {attr}. Skipping. {e}"
                )
        return convert_to_json_friendly(attributes)


def save_as_pickle_gzip(obj, filepath):
    pickled_data = pickle.dumps(obj)
    save_as_gzip(pickled_data, filepath)


def save_as_json_gzip(obj, filepath):
    json_data = json.dumps(
        convert_to_json_friendly(obj), separators=(",", ":")
    )
    save_as_gzip(json_data.encode("utf-8"), filepath)


def save_as_gzip(data, filepath, mode="wb"):
    if os.path.exists(filepath):
        raise FileExistsError(filepath)
    if len(Path(filepath).parents) > 0:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(filepath, mode) as file:
        file.write(data)
    print("wrote " + filepath)


def load_pickle_gzip(filepath):
    with gzip.open(filepath, "rb") as file:
        loaded_object = pickle.load(file)
    return loaded_object


def load_json_gzip(filepath):
    with gzip.open(filepath, "rb") as file:
        json_data = file.read().decode("utf-8")
        loaded_data = json.loads(json_data)
    return loaded_data


class NullRecorder:
    def record(self, key, value):
        pass

    def get_nested_recorder(self, key):
        return NullRecorder()


class BaseRecorder:
    @abstractmethod
    def _get_this_dict(self):
        pass

    def record(self, key, value):
        this_dict = self._get_this_dict()
        assert key not in this_dict
        this_dict[key] = value

    def get_nested_recorder(self, key):
        return NestedRecorder(self, key)

    def _get_nested_dict(self, key):
        this_dict = self._get_this_dict()
        if key in this_dict:
            assert isinstance(this_dict[key], dict)
        else:
            this_dict[key] = {}

        return this_dict[key]


class StepRecorder(BaseRecorder):
    def __init__(self):
        self._partial_step_dict: Any = {}
        self._steps: List[Any] = []

    def finish_step(self):
        self._steps.append(self._partial_step_dict)
        self._partial_step_dict = {}

    def reset(self):
        self._partial_step_dict = {}
        self._steps = []

    def _get_this_dict(self):
        return self._partial_step_dict


class NestedRecorder(BaseRecorder):
    def __init__(self, parent, scope_key):
        self._parent = parent
        self._scope_key = scope_key

    def _get_this_dict(self):
        return self._parent._get_nested_dict(self._scope_key)
