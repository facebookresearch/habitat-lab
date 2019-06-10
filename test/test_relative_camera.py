import os

import pytest

import habitat
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1


def test_demo_notebook():
    config = habitat.get_config("configs/tasks/pointnav_rgbd.yaml")
    config.defrost()
    config.DATASET.SPLIT = "val"

    if not PointNavDatasetV1.check_config_paths_exist(config.DATASET):
        pytest.skip(
            "Please download the habitat test scenes"
        )
    else:
        pytest.main(
            [
                "--nbval-lax",
                "notebooks/relative_camera_views_transform_and_warping_demo.ipynb",
            ]
        )
