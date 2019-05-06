#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time

import pytest

import habitat
from habitat.config.default import get_config
from habitat.core.embodied_task import Episode
from habitat.core.logging import logger
from habitat.datasets import make_dataset
from habitat.datasets.pointnav.pointnav_dataset import (
    PointNavDatasetV1,
    DEFAULT_SCENE_PATH_PREFIX,
)

CFG_TEST = "configs/datasets/pointnav/gibson.yaml"
PARTIAL_LOAD_SCENES = 3


def check_json_serializaiton(dataset: habitat.Dataset):
    start_time = time.time()
    json_str = str(dataset.to_json())
    logger.info(
        "JSON conversion finished. {} sec".format((time.time() - start_time))
    )
    decoded_dataset = dataset.__class__()
    decoded_dataset.from_json(json_str)
    assert len(decoded_dataset.episodes) > 0
    episode = decoded_dataset.episodes[0]
    assert isinstance(episode, Episode)
    assert (
        decoded_dataset.to_json() == json_str
    ), "JSON dataset encoding/decoding isn't consistent"


def test_single_pointnav_dataset():
    dataset_config = get_config().DATASET
    if not PointNavDatasetV1.check_config_paths_exist(dataset_config):
        pytest.skip("Test skipped as dataset files are missing.")
    scenes = PointNavDatasetV1.get_scenes_to_load(config=dataset_config)
    assert (
        len(scenes) == 0
    ), "Expected dataset doesn't expect separate episode file per scene."
    dataset = PointNavDatasetV1(config=dataset_config)
    assert len(dataset.episodes) > 0, "The dataset shouldn't be empty."
    assert (
        len(dataset.scene_ids) == 2
    ), "The test dataset scenes number is wrong."
    check_json_serializaiton(dataset)


def test_multiple_files_scene_path():
    dataset_config = get_config(CFG_TEST).DATASET
    if not PointNavDatasetV1.check_config_paths_exist(dataset_config):
        pytest.skip("Test skipped as dataset files are missing.")
    scenes = PointNavDatasetV1.get_scenes_to_load(config=dataset_config)
    assert (
        len(scenes) > 0
    ), "Expected dataset contains separate episode file per scene."
    dataset_config.defrost()
    dataset_config.POINTNAVV1.CONTENT_SCENES = scenes[:PARTIAL_LOAD_SCENES]
    dataset_config.SCENES_DIR = os.path.join(
        os.getcwd(), DEFAULT_SCENE_PATH_PREFIX
    )
    dataset_config.freeze()
    partial_dataset = make_dataset(
        id_dataset=dataset_config.TYPE, config=dataset_config
    )
    assert (
        len(partial_dataset.scene_ids) == PARTIAL_LOAD_SCENES
    ), "Number of loaded scenes doesn't correspond."
    print(partial_dataset.episodes[0].scene_id)
    assert os.path.exists(
        partial_dataset.episodes[0].scene_id
    ), "Scene file {} doesn't exist using absolute path".format(
        partial_dataset.episodes[0].scene_id
    )


def test_multiple_files_pointnav_dataset():
    dataset_config = get_config(CFG_TEST).DATASET
    if not PointNavDatasetV1.check_config_paths_exist(dataset_config):
        pytest.skip("Test skipped as dataset files are missing.")
    scenes = PointNavDatasetV1.get_scenes_to_load(config=dataset_config)
    assert (
        len(scenes) > 0
    ), "Expected dataset contains separate episode file per scene."
    dataset_config.defrost()
    dataset_config.POINTNAVV1.CONTENT_SCENES = scenes[:PARTIAL_LOAD_SCENES]
    dataset_config.freeze()
    partial_dataset = make_dataset(
        id_dataset=dataset_config.TYPE, config=dataset_config
    )
    assert (
        len(partial_dataset.scene_ids) == PARTIAL_LOAD_SCENES
    ), "Number of loaded scenes doesn't correspond."
    check_json_serializaiton(partial_dataset)
