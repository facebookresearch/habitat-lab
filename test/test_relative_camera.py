#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import gc

import pytest

import habitat
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1

try:
    import torch  # noqa # pylint: disable=unused-import

    has_torch = True
except ImportError:
    has_torch = False


@pytest.mark.skipif(not has_torch, reason="Test needs torch")
def test_demo_notebook():
    config = habitat.get_config("configs/tasks/pointnav_rgbd.yaml")
    config.defrost()
    config.DATASET.SPLIT = "val"

    if not PointNavDatasetV1.check_config_paths_exist(config.DATASET):
        pytest.skip("Please download the habitat test scenes")
    else:
        pytest.main(
            [
                "--nbval-lax",
                "notebooks/relative_camera_views_transform_and_warping_demo.ipynb",
            ]
        )

        # NB: Force a gc collect run as it can take a little bit for
        # the cleanup to happen after the notebook and we get
        # a double context crash!
        gc.collect()
