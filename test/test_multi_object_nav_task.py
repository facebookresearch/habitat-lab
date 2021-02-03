#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
import os
import time

import pytest

import habitat
from habitat.config.default import get_config
from habitat.core.embodied_task import Episode
from habitat.core.logging import logger
from habitat.datasets import make_dataset
from habitat.datasets.multi_object_nav.multi_object_nav_dataset import (
    MultiObjectNavDatasetV1,
)
from habitat.tasks.nav.nav import MoveForwardAction

CFG_TEST = "configs/test/habitat_mp3d_multi_object_nav_test.yaml"
EPISODES_LIMIT = 6
PARTIAL_LOAD_SCENES = 3


def check_json_serializaiton(dataset: habitat.Dataset):
    start_time = time.time()
    json_str = dataset.to_json()
    logger.info(
        "JSON conversion finished. {} sec".format((time.time() - start_time))
    )
    decoded_dataset = MultiObjectNavDatasetV1()
    decoded_dataset.from_json(json_str)
    assert len(decoded_dataset.episodes) == len(dataset.episodes)
    episode = decoded_dataset.episodes[0]
    assert isinstance(episode, Episode)

    assert json.loads(decoded_dataset.to_json()) == json.loads(
        json_str
    ), "JSON dataset encoding/decoding isn't consistent"


def test_mp3d_multi_object_nav_dataset():
    dataset_config = get_config(CFG_TEST).DATASET
    if not MultiObjectNavDatasetV1.check_config_paths_exist(dataset_config):
        pytest.skip(
            "Please download Matterport3D MultiON Dataset to data folder."
        )

    dataset = habitat.make_dataset(
        id_dataset=dataset_config.TYPE, config=dataset_config
    )
    assert dataset
    dataset.episodes = dataset.episodes[0:EPISODES_LIMIT]
    check_json_serializaiton(dataset)


@pytest.mark.parametrize("split", ["train", "val"])
def test_dataset_splitting(split):
    dataset_config = get_config(CFG_TEST).DATASET
    dataset_config.defrost()
    dataset_config.SPLIT = split

    if not MultiObjectNavDatasetV1.check_config_paths_exist(dataset_config):
        pytest.skip("Test skipped as dataset files are missing.")

    scenes = MultiObjectNavDatasetV1.get_scenes_to_load(config=dataset_config)
    assert (
        len(scenes) > 0
    ), "Expected dataset contains separate episode file per scene."

    dataset_config.CONTENT_SCENES = scenes[:PARTIAL_LOAD_SCENES]
    full_dataset = make_dataset(
        id_dataset=dataset_config.TYPE, config=dataset_config
    )
    full_episodes = {
        (ep.scene_id, ep.episode_id) for ep in full_dataset.episodes
    }

    dataset_config.CONTENT_SCENES = scenes[: PARTIAL_LOAD_SCENES // 2]
    split1_dataset = make_dataset(
        id_dataset=dataset_config.TYPE, config=dataset_config
    )
    split1_episodes = {
        (ep.scene_id, ep.episode_id) for ep in split1_dataset.episodes
    }

    dataset_config.CONTENT_SCENES = scenes[
        PARTIAL_LOAD_SCENES // 2 : PARTIAL_LOAD_SCENES
    ]
    split2_dataset = make_dataset(
        id_dataset=dataset_config.TYPE, config=dataset_config
    )
    split2_episodes = {
        (ep.scene_id, ep.episode_id) for ep in split2_dataset.episodes
    }

    assert full_episodes == split1_episodes.union(
        split2_episodes
    ), "Split dataset is not equal to full dataset"
    assert (
        len(split1_episodes.intersection(split2_episodes)) == 0
    ), "Intersection of split datasets is not the empty set"


def test_multi_object_nav_task():
    config = get_config(CFG_TEST)

    if not MultiObjectNavDatasetV1.check_config_paths_exist(config.DATASET):
        pytest.skip(
            "Please download Matterport3D scene and MultiON Datasets to data folder."
        )

    dataset = make_dataset(
        id_dataset=config.DATASET.TYPE, config=config.DATASET
    )
    with habitat.Env(config=config, dataset=dataset) as env:
        if not os.path.exists(config.TASK.OBJECTS_PATH):
            pytest.skip("Please download MultiON objects in the data folder.")
        for _ in range(10):
            env.reset()
            while not env.episode_over:
                action = env.action_space.sample()
                habitat.logger.info(
                    f"Action : "
                    f"{action['action']}, "
                    f"args: {action['action_args']}."
                )
                env.step(action)

            metrics = env.get_metrics()
            logger.info(metrics)

        with pytest.raises(AssertionError):
            env.step({"action": MoveForwardAction.name})
