#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import time

import numpy as np
import pytest

import habitat
from habitat.config.default import get_config
from habitat.core.embodied_task import Episode
from habitat.core.logging import logger
from habitat.datasets import make_dataset
from habitat.datasets.pointnav import pointnav_generator as pointnav_generator
from habitat.datasets.pointnav.pointnav_dataset import (
    DEFAULT_SCENE_PATH_PREFIX,
    PointNavDatasetV1,
)
from habitat.utils.geometry_utils import (
    angle_between_quaternions,
    quaternion_from_coeff,
)

CFG_TEST = "test/config/habitat/habitat_all_sensors_test.yaml"
CFG_MULTI_TEST = "benchmark/nav/pointnav/pointnav_gibson.yaml"
PARTIAL_LOAD_SCENES = 3
NUM_EPISODES = 10


def check_json_serialization(dataset: habitat.Dataset):
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
    dataset_config = get_config(
        "benchmark/nav/pointnav/pointnav_habitat_test.yaml"
    ).habitat.dataset
    if not PointNavDatasetV1.check_config_paths_exist(dataset_config):
        pytest.skip("Test skipped as dataset files are missing.")
    scenes = PointNavDatasetV1.get_scenes_to_load(config=dataset_config)
    assert (
        len(scenes) > 0
    ), "Expected dataset contains separate episode file per scene."
    dataset = PointNavDatasetV1(config=dataset_config)
    assert len(dataset.episodes) > 0, "The dataset shouldn't be empty."
    assert (
        len(dataset.scene_ids) == 2
    ), "The test dataset scenes number is wrong."
    check_json_serialization(dataset)


def test_multiple_files_scene_path():
    dataset_config = get_config(CFG_MULTI_TEST).habitat.dataset
    if not PointNavDatasetV1.check_config_paths_exist(dataset_config):
        pytest.skip("Test skipped as dataset files are missing.")
    scenes = PointNavDatasetV1.get_scenes_to_load(config=dataset_config)
    assert (
        len(scenes) > 0
    ), "Expected dataset contains separate episode file per scene."
    with habitat.config.read_write(dataset_config):
        dataset_config.content_scenes = scenes[:PARTIAL_LOAD_SCENES]
        dataset_config.scenes_dir = os.path.join(
            os.getcwd(), DEFAULT_SCENE_PATH_PREFIX
        )
    partial_dataset = make_dataset(
        id_dataset=dataset_config.type, config=dataset_config
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
    dataset_config = get_config(CFG_MULTI_TEST).habitat.dataset
    if not PointNavDatasetV1.check_config_paths_exist(dataset_config):
        pytest.skip("Test skipped as dataset files are missing.")
    scenes = PointNavDatasetV1.get_scenes_to_load(config=dataset_config)
    assert (
        len(scenes) > 0
    ), "Expected dataset contains separate episode file per scene."
    with habitat.config.read_write(dataset_config):
        dataset_config.content_scenes = scenes[:PARTIAL_LOAD_SCENES]
    partial_dataset = make_dataset(
        id_dataset=dataset_config.type, config=dataset_config
    )
    assert (
        len(partial_dataset.scene_ids) == PARTIAL_LOAD_SCENES
    ), "Number of loaded scenes doesn't correspond."
    check_json_serialization(partial_dataset)


@pytest.mark.parametrize("split", ["train", "val"])
def test_dataset_splitting(split):
    dataset_config = get_config(CFG_MULTI_TEST).habitat.dataset
    with habitat.config.read_write(dataset_config):
        dataset_config.split = split

        if not PointNavDatasetV1.check_config_paths_exist(dataset_config):
            pytest.skip("Test skipped as dataset files are missing.")

        scenes = PointNavDatasetV1.get_scenes_to_load(config=dataset_config)
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


def check_shortest_path(env, episode):
    def check_state(agent_state, position, rotation):
        assert (
            angle_between_quaternions(
                agent_state.rotation, quaternion_from_coeff(rotation)
            )
            < 1e-5
        ), "Agent's rotation diverges from the shortest path."

        assert np.allclose(
            agent_state.position, position
        ), "Agent's position position diverges from the shortest path's one."

    assert len(episode.goals) == 1, "Episode has no goals or more than one."
    assert (
        len(episode.shortest_paths) == 1
    ), "Episode has no shortest paths or more than one."

    env.episode_iterator = iter([episode])
    env.reset()
    start_state = env.sim.get_agent_state()
    check_state(start_state, episode.start_position, episode.start_rotation)

    for point in episode.shortest_paths[0]:
        cur_state = env.sim.get_agent_state()
        check_state(cur_state, point.position, point.rotation)
        env.step(point.action)


def test_pointnav_episode_generator():
    config = get_config(CFG_TEST)
    with habitat.config.read_write(config):
        config.habitat.dataset.split = "val"
        config.habitat.environment.max_episode_steps = 500
    if not PointNavDatasetV1.check_config_paths_exist(config.habitat.dataset):
        pytest.skip("Test skipped as dataset files are missing.")
    with habitat.Env(config) as env:
        env.seed(config.habitat.seed)
        random.seed(config.habitat.seed)
        generator = pointnav_generator.generate_pointnav_episode(
            sim=env.sim,
            shortest_path_success_distance=config.habitat.task.measurements.success.success_distance,
            shortest_path_max_steps=config.habitat.environment.max_episode_steps,
        )
        episodes = []
        for _ in range(NUM_EPISODES):
            episode = next(generator)
            episodes.append(episode)

        for episode in pointnav_generator.generate_pointnav_episode(
            sim=env.sim,
            num_episodes=NUM_EPISODES,
            shortest_path_success_distance=config.habitat.task.measurements.success.success_distance,
            shortest_path_max_steps=config.habitat.environment.max_episode_steps,
            geodesic_to_euclid_min_ratio=0,
        ):
            episodes.append(episode)

        assert len(episodes) == 2 * NUM_EPISODES
        env.episode_iterator = iter(episodes)

        for episode in episodes:
            check_shortest_path(env, episode)

        dataset: habitat.Dataset = habitat.Dataset()
        dataset.episodes = episodes
        assert (
            dataset.to_json()
        ), "Generated episodes aren't json serializable."
