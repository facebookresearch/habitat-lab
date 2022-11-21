#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import gc

import pytest

import habitat
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1


def test_demo_notebook():
    config = habitat.get_config("benchmark/nav/pointnav/pointnav_mp3d.yaml")
    with habitat.config.read_write(config):
        config.habitat.dataset.split = "val"

    if not PointNavDatasetV1.check_config_paths_exist(config.habitat.dataset):
        pytest.skip(
            "Please download the Matterport3D PointNav val dataset and Matterport3D val scenes"
        )
    else:
        pytest.main(["--nbval-lax", "notebooks/habitat-lab-demo.ipynb"])

        # NB: Force a gc collect run as it can take a little bit for
        # the cleanup to happen after the notebook and we get
        # a double context crash!
        gc.collect()
