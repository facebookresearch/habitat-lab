#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import time

import pytest

import habitat
import habitat.tasks.rearrange.rearrange_sim
import habitat.tasks.rearrange.rearrange_task
import habitat_baselines.utils.env_utils
from habitat.config.default import get_config
from habitat.core.embodied_task import Episode
from habitat.core.logging import logger
from habitat.datasets import make_dataset
from habitat.datasets.rearrange.rearrange_dataset import RearrangeDatasetV0
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.config.default import get_config as baselines_get_config

CFG_TEST = "configs/tasks/rearrangepick_replica_cad.yaml"
EPISODES_LIMIT = 6
PARTIAL_LOAD_SCENES = 3


def check_json_serializaiton(dataset: habitat.Dataset):
    start_time = time.time()
    json_str = dataset.to_json()
    logger.info(
        "JSON conversion finished. {} sec".format((time.time() - start_time))
    )
    decoded_dataset = RearrangeDatasetV0()
    decoded_dataset.from_json(json_str)
    decoded_dataset.config = dataset.config
    assert len(decoded_dataset.episodes) == len(dataset.episodes)
    episode = decoded_dataset.episodes[0]
    assert isinstance(episode, Episode)

    # The strings won't match exactly as dictionaries don't have an order for the keys
    # Thus we need to parse the json strings and compare the serialized forms
    assert json.loads(decoded_dataset.to_json()) == json.loads(
        json_str
    ), "JSON dataset encoding/decoding isn't consistent"


def test_rearrange_dataset():
    dataset_config = get_config(CFG_TEST).DATASET
    if not RearrangeDatasetV0.check_config_paths_exist(dataset_config):
        pytest.skip(
            "Please download ReplicaCAD RearrangeDataset Dataset to data folder."
        )

    dataset = habitat.make_dataset(
        id_dataset=dataset_config.TYPE, config=dataset_config
    )
    assert dataset
    dataset.episodes = dataset.episodes[0:EPISODES_LIMIT]
    check_json_serializaiton(dataset)


@pytest.mark.parametrize("split", ["train", "test"])
def test_dataset_splitting(split):
    dataset_config = get_config(CFG_TEST).DATASET
    dataset_config.defrost()
    dataset_config.SPLIT = split

    if not RearrangeDatasetV0.check_config_paths_exist(dataset_config):
        pytest.skip("Test skipped as dataset files are missing.")

    scenes = RearrangeDatasetV0.get_scenes_to_load(config=dataset_config)
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


def test_rearrange_habitat_env():
    config = get_config(CFG_TEST)

    if not RearrangeDatasetV0.check_config_paths_exist(config.DATASET):
        pytest.skip("Test skipped as dataset files are missing.")

    config.freeze()
    with habitat.Env(config=config, dataset=None) as env:
        for _ in range(10):
            env.reset()
            while not env.episode_over:
                action = env.action_space.sample()
                obs = env.step(action)
                habitat.logger.info(
                    f"Action : "
                    f"{action['action']}, "
                    f"args: {action['action_args']}."
                    f"obs: {list(obs.keys())}."
                )

        env.reset()


def test_rearrange_task():
    config = baselines_get_config(
        "habitat_baselines/config/rearrange/ddppo_rearrangepick.yaml"
    )
    # if not RearrangeDatasetV0.check_config_paths_exist(config.TASK_CONFIG.DATASET):
    #     pytest.skip("Test skipped as dataset files are missing.")

    env_class = get_env_class(config.ENV_NAME)

    env = habitat_baselines.utils.env_utils.make_env_fn(
        env_class=env_class, config=config
    )

    with env:
        for _ in range(10):
            env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                habitat.logger.info(
                    f"Action : "
                    f"{action['action']}, "
                    f"args: {action['action_args']}."
                )
                _, _, done, info = env.step(action=action)

            logger.info(info)
