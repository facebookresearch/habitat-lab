import os

import pytest

import habitat
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1


def test_demo_notebook():
    config = habitat.get_config("configs/tasks/pointnav_mp3d.yaml")
    config.defrost()
    config.DATASET.SPLIT = "val"

    if not PointNavDatasetV1.check_config_paths_exist(config.DATASET):
        pytest.skip(
            "Please download the Matterport3D PointNav val dataset and Matterport3D val scenes"
        )
    else:
        pytest.main(["--nbval-lax", "notebooks/habitat-api-demo.ipynb"])
