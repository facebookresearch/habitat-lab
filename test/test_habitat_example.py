#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import habitat
from examples.example import example
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1


def test_readme_example():
    if not PointNavDatasetV1.check_config_paths_exist(
        config=habitat.get_config().DATASET
    ):
        pytest.skip("Please download Habitat test data to data folder.")
    example()
