#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

import pytest

import habitat
import habitat.datasets.vln.r2r_vln_dataset as r2r_vln_dataset
from habitat.config.default import get_config
from habitat.core.logging import logger
from habitat.datasets import make_dataset
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.tasks.vln.vln import VLNEpisode

CFG_TEST = "configs/test/habitat_r2r_vln_test.yaml"
R2R_VAL_SEEN_EPISODES = 778
EPISODES_LIMIT = 1


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
    assert isinstance(episode, VLNEpisode)
    assert (
        decoded_dataset.to_json() == json_str
    ), "JSON dataset encoding/decoding isn't consistent"


def test_r2r_vln_dataset():
    vln_config = get_config(CFG_TEST)
    if not r2r_vln_dataset.VLNDatasetV1.check_config_paths_exist(
        vln_config.DATASET
    ):
        pytest.skip("Please download Matterport3D R2R dataset to data folder.")

    dataset = make_dataset(
        id_dataset=vln_config.DATASET.TYPE, config=vln_config.DATASET
    )
    assert dataset
    assert (
        len(dataset.episodes) == R2R_VAL_SEEN_EPISODES
    ), "Val Seen split episode number mismatch"
    check_json_serializaiton(dataset)


def test_r2r_vln_sim():
    vln_config = get_config(CFG_TEST)

    if not r2r_vln_dataset.VLNDatasetV1.check_config_paths_exist(
        vln_config.DATASET
    ):
        pytest.skip(
            "Please download Matterport3D R2R VLN dataset to data folder."
        )

    dataset = make_dataset(
        id_dataset=vln_config.DATASET.TYPE, config=vln_config.DATASET
    )

    env = habitat.Env(config=vln_config, dataset=dataset)
    env.episodes = dataset.episodes[:EPISODES_LIMIT]

    follower = ShortestPathFollower(
        env.sim, goal_radius=0.5, return_one_hot=False
    )
    assert env

    for i in range(len(env.episodes)):
        env.reset()
        path = env.current_episode.reference_path + [
            env.current_episode.goals[0].position
        ]
        for point in path:
            done = False
            while not done:
                best_action = follower.get_next_action(point)
                if best_action == None:
                    break
                obs = env.step(best_action)
                assert "rgb" in obs, "RGB image is missing in observation."
                assert (
                    "instruction" in obs
                ), "Instruction is missing in observation."
                assert (
                    obs["instruction"]["text"]
                    == env.current_episode.instruction.instruction_text
                ), "Instruction from sensor does not match the intruction from the episode"

                assert obs["rgb"].shape[:2] == (
                    vln_config.SIMULATOR.RGB_SENSOR.HEIGHT,
                    vln_config.SIMULATOR.RGB_SENSOR.WIDTH,
                ), (
                    "Observation resolution {} doesn't correspond to config "
                    "({}, {}).".format(
                        obs["rgb"].shape[:2],
                        vln_config.SIMULATOR.RGB_SENSOR.HEIGHT,
                        vln_config.SIMULATOR.RGB_SENSOR.WIDTH,
                    )
                )

    env.close()
