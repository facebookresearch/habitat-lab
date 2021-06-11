#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys

import mock
import numpy as np

from habitat.config.default import get_config
from habitat.sims import make_sim


class CameraMock:
    def get_rgb(self):
        return np.zeros((256, 256, 3))

    def get_depth(self):
        return np.zeros((256, 256, 1))

    def reset(self):
        pass

    def step(self, *args, **kwargs):
        pass


class RobotMock:  # noqa: SIM119
    def __init__(self, *args, **kwargs):
        self.camera = CameraMock()
        self.base = BaseMock()


class BaseMock:
    def __init__(self, *args, **kwargs):
        self.base_state = mock.MagicMock()
        self.base_state.bumper = False

    def go_to_relative(self, *args, **kwargs):
        pass


def test_pyrobot(mocker):
    if "pyrobot" not in sys.modules:
        # Mock pyrobot package if it is not installed
        mock_pyrobot = mocker.MagicMock()
        mock_pyrobot.Robot = RobotMock
        sys.modules["pyrobot"] = mock_pyrobot

        # Re-register pyrobot with mock
        from habitat.sims.registration import _try_register_pyrobot

        _try_register_pyrobot()

    config = get_config()
    with make_sim("PyRobot-v0", config=config.PYROBOT) as reality:

        _ = reality.reset()
        _ = reality.step(
            "go_to_relative",
            {
                "xyt_position": [0, 0, (10 / 180) * np.pi],
                "use_map": False,
                "close_loop": True,
                "smooth": False,
            },
        )
