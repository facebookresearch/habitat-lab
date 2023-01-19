#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import habitat
from examples import (
    new_actions,
    register_new_sensors_and_measures,
    shortest_path_follower_example,
)
from examples.example import example
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1


def skip_if_dataset_does_not_exist():
    cfg = habitat.get_config(
        "benchmark/nav/pointnav/pointnav_habitat_test.yaml"
    )
    if not PointNavDatasetV1.check_config_paths_exist(
        config=cfg.habitat.dataset
    ):
        pytest.skip("Please download Habitat test data to data folder.")


def test_readme_example():
    skip_if_dataset_does_not_exist()
    example()


def test_shortest_path_follower_example():
    skip_if_dataset_does_not_exist()
    shortest_path_follower_example.main()


def test_register_new_sensors_and_measures():
    skip_if_dataset_does_not_exist()
    register_new_sensors_and_measures.main()


def test_new_actions():
    skip_if_dataset_does_not_exist()
    new_actions.main()
