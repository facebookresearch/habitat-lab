#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.registry import registry
from habitat.core.simulator import Simulator


def _try_register_pyrobot():
    try:
        import pyrobot  # noqa: F401

        has_pyrobot = True
    except ImportError as e:
        has_pyrobot = False
        pyrobot_import_error = e

    if has_pyrobot:
        from habitat.sims.pyrobot.pyrobot import PyRobot  # noqa: F401
    else:

        @registry.register_simulator(name="PyRobot-v0")
        class PyRobotImportError(Simulator):
            def __init__(self, *args, **kwargs):
                raise pyrobot_import_error
