#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import habitat
from examples import (
    new_actions,
    register_new_sensors_and_measures,
    shortest_path_follower_example,
    visualization_examples,
)
from examples.example import example
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1


@pytest.mark.parametrize(
    "example_fun_fn",
    [
        example,
        new_actions.main,
        register_new_sensors_and_measures.main,
        shortest_path_follower_example.main,
        visualization_examples.main,
    ],
)
def test_examples(example_fun_fn):
    if not PointNavDatasetV1.check_config_paths_exist(
        config=habitat.get_config().habitat.dataset
    ):
        pytest.skip("Please download Habitat test data to data folder.")

    example_fun_fn()
