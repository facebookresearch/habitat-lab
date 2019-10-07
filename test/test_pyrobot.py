#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import importlib
import os
import sys


def init_pyrobot():
    from habitat.sims import make_sim
    from habitat.config.default import get_config
    config = get_config()

    reality = make_sim("PyRobot-v0", config=config.PYROBOT)


def test_pyrobot(mocker):
    if "pyrobot" not in sys.modules:
        # Mock pyrobot package if it is not installed
        sys.modules["pyrobot"] = mocker.MagicMock()

        # Re-register pyrobot with mock
        from habitat.sims.registration import _try_register_pyrobot
        _try_register_pyrobot()

    init_pyrobot()
