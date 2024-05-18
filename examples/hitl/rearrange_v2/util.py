#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from time import time

import magnum as mn

UP = mn.Vector3(0, 1, 0)
FWD = mn.Vector3(0, 0, 1)


def timestamp() -> str:
    "Generate a Unix timestamp at the current time."
    return str(int(time()))


def get_empty_view(sim) -> mn.Matrix4:
    """
    Get a view looking into the void.
    Used to avoid displaying previously-loaded content in intermediate stages.
    """
    return mn.Matrix4.look_at(1000 * FWD, FWD, UP)
