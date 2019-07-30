#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from glob import glob

import pytest

try:
    from habitat_baselines.run import run_exp

    baseline_installed = True
except ImportError:
    baseline_installed = False


@pytest.mark.skipif(
    not baseline_installed, reason="baseline sub-module not installed"
)
@pytest.mark.parametrize(
    "test_cfg_path,mode",
    [
        (cfg, mode)
        for mode in ("train", "eval")
        for cfg in glob("habitat_baselines/config/test/*")
    ],
)
def test_trainers(test_cfg_path, mode):
    run_exp(test_cfg_path, mode)
