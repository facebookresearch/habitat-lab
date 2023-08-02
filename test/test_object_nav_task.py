#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import time

import pytest

import habitat
from habitat.config.default import get_config
from habitat.core.embodied_task import Episode
from habitat.core.logging import logger
from habitat.datasets import make_dataset
from habitat.datasets.object_nav.object_nav_dataset import ObjectNavDatasetV1
from habitat.tasks.nav.nav import MoveForwardAction

EPISODES_LIMIT = 6
PARTIAL_LOAD_SCENES = 3


def check_json_serialization(dataset: habitat.Dataset):
    start_time = time.time()
    json_str = dataset.to_json()
    logger.info(
        "JSON conversion finished. {} sec".format((time.time() - start_time))
    )
    decoded_dataset = ObjectNavDatasetV1()
    decoded_dataset.from_json(json_str)
    assert len(decoded_dataset.episodes) == len(dataset.episodes)
    episode = decoded_dataset.episodes[0]
    assert isinstance(episode, Episode)

    # The strings won't match exactly as dictionaries don't have an order for the keys
    # Thus we need to parse the json strings and compare the serialized forms
    assert json.loads(decoded_dataset.to_json()) == json.loads(
        json_str
    ), "JSON dataset encoding/decoding isn't consistent"


@pytest.mark.parametrize(
    "config_file",
    [
        "test/config/habitat/habitat_mp3d_object_nav_test.yaml",
        "test/config/habitat/habitat_hssd-hab_object_nav_test.yaml",
        "test/config/habitat/habitat_procthor-hab_object_nav_test.yaml",
    ],
)
def test_object_nav_dataset(config_file):
    dataset_config = get_config(config_file).habitat.dataset
    if not ObjectNavDatasetV1.check_config_paths_exist(dataset_config):
        pytest.skip("Please download ObjectNav dataset to data folder.")

    dataset = habitat.make_dataset(
        id_dataset=dataset_config.type, config=dataset_config
    )
    assert dataset
    dataset.episodes = dataset.episodes[0:EPISODES_LIMIT]
    dataset.goals_by_category = {
        k: v
        for k, v in dataset.goals_by_category.items()
        if k in (ep.goals_key for ep in dataset.episodes)
    }
    check_json_serialization(dataset)


@pytest.mark.parametrize(
    "config_file",
    [
        "test/config/habitat/habitat_mp3d_object_nav_test.yaml",
        "test/config/habitat/habitat_hssd-hab_object_nav_test.yaml",
        "test/config/habitat/habitat_procthor-hab_object_nav_test.yaml",
    ],
)
@pytest.mark.parametrize("split", ["train", "val"])
def test_dataset_splitting(config_file, split):
    dataset_config = get_config(config_file).habitat.dataset
    with habitat.config.read_write(dataset_config):
        dataset_config.split = split

        if not ObjectNavDatasetV1.check_config_paths_exist(dataset_config):
            pytest.skip("Test skipped as dataset files are missing.")

        scenes = ObjectNavDatasetV1.get_scenes_to_load(config=dataset_config)
        assert (
            len(scenes) > 0
        ), "Expected dataset contains separate episode file per scene."

        dataset_config.content_scenes = scenes[:PARTIAL_LOAD_SCENES]
        full_dataset = make_dataset(
            id_dataset=dataset_config.type, config=dataset_config
        )
        full_episodes = {
            (ep.scene_id, ep.episode_id) for ep in full_dataset.episodes
        }

        dataset_config.content_scenes = scenes[: PARTIAL_LOAD_SCENES // 2]
        split1_dataset = make_dataset(
            id_dataset=dataset_config.type, config=dataset_config
        )
        split1_episodes = {
            (ep.scene_id, ep.episode_id) for ep in split1_dataset.episodes
        }

        dataset_config.content_scenes = scenes[
            PARTIAL_LOAD_SCENES // 2 : PARTIAL_LOAD_SCENES
        ]
        split2_dataset = make_dataset(
            id_dataset=dataset_config.type, config=dataset_config
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


@pytest.mark.parametrize(
    "config_file",
    [
        "test/config/habitat/habitat_mp3d_object_nav_test.yaml",
        "test/config/habitat/habitat_hssd-hab_object_nav_test.yaml",
        "test/config/habitat/habitat_procthor-hab_object_nav_test.yaml",
    ],
)
def test_object_nav_task(config_file):
    config = get_config(config_file)

    if not ObjectNavDatasetV1.check_config_paths_exist(config.habitat.dataset):
        pytest.skip(
            "Please download scene and ObjectNav dataset to data folder."
        )

    dataset = make_dataset(
        id_dataset=config.habitat.dataset.type, config=config.habitat.dataset
    )
    with habitat.Env(config=config, dataset=dataset) as env:
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
