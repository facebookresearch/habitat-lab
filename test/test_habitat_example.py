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


def test_readme_example():
    if not PointNavDatasetV1.check_config_paths_exist(
        config=habitat.get_config().DATASET
    ):
        pytest.skip("Please download Habitat test data to data folder.")
    example()


def test_visualizations_example():
    if not PointNavDatasetV1.check_config_paths_exist(
        config=habitat.get_config().DATASET
    ):
        pytest.skip("Please download Habitat test data to data folder.")
    visualization_examples.main()


def test_shortest_path_follower_example():
    if not PointNavDatasetV1.check_config_paths_exist(
        config=habitat.get_config().DATASET
    ):
        pytest.skip("Please download Habitat test data to data folder.")
    shortest_path_follower_example.main()


def test_register_new_sensors_and_measures():
    if not PointNavDatasetV1.check_config_paths_exist(
        config=habitat.get_config().DATASET
    ):
        pytest.skip("Please download Habitat test data to data folder.")

    register_new_sensors_and_measures.main()


def test_new_actions():
    if not PointNavDatasetV1.check_config_paths_exist(
        config=habitat.get_config().DATASET
    ):
        pytest.skip("Please download Habitat test data to data folder.")

    new_actions.main()
