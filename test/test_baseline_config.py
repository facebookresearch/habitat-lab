#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from glob import glob

import pytest

from habitat_baselines.config.default import get_config


@pytest.mark.parametrize(
    "test_cfg_path",
    list(
        glob(
            "habitat-baselines/habitat_baselines/config/**/*.yaml",
            recursive=True,
        ),
    ),
)
def test_baselines_configs(test_cfg_path):
    cleaned_path = test_cfg_path.replace(
        "habitat-baselines/habitat_baselines/config/", ""
    )
    if "habitat_baselines" in cleaned_path:
        # Do not test non-standalone config options that are
        # supposed to be used with "main" configs.
        return

    get_config(cleaned_path)
