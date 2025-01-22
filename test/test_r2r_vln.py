#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

import pytest

import habitat
from habitat.config.default import get_agent_config, get_config
from habitat.core.logging import logger
from habitat.datasets import make_dataset
from habitat.datasets.vln import r2r_vln_dataset as r2r_vln_dataset
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.tasks.vln.vln import VLNEpisode

CFG_TEST = "test/config/habitat/habitat_r2r_vln_test.yaml"
R2R_VAL_SEEN_EPISODES = 778
EPISODES_LIMIT = 1


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
    assert isinstance(episode, VLNEpisode)
    assert (
        decoded_dataset.to_json() == json_str
    ), "JSON dataset encoding/decoding isn't consistent"


def test_r2r_vln_dataset():
    vln_config = get_config(CFG_TEST)
    if not r2r_vln_dataset.VLNDatasetV1.check_config_paths_exist(
        vln_config.habitat.dataset
    ):
        pytest.skip("Please download Matterport3D R2R dataset to data folder.")

    dataset = make_dataset(
        id_dataset=vln_config.habitat.dataset.type,
        config=vln_config.habitat.dataset,
    )
    assert dataset
    assert (
        len(dataset.episodes) == R2R_VAL_SEEN_EPISODES
    ), "Val Seen split episode number mismatch"

    check_json_serialization(dataset)


@pytest.mark.parametrize("split", ["train", "val_seen", "val_unseen"])
def test_dataset_splitting(split):
    dataset_config = get_config(CFG_TEST).habitat.dataset
    with habitat.config.read_write(dataset_config):
        dataset_config.split = split

        if not r2r_vln_dataset.VLNDatasetV1.check_config_paths_exist(
            dataset_config
        ):
            pytest.skip(
                "Please download Matterport3D R2R dataset to data folder."
            )

        scenes = r2r_vln_dataset.VLNDatasetV1.get_scenes_to_load(
            config=dataset_config
        )
        assert (
            len(scenes) > 0
        ), "Expected dataset contains separate episode file per scene."

        dataset_config.content_scenes = scenes
        full_dataset = make_dataset(
            id_dataset=dataset_config.type, config=dataset_config
        )
        full_episodes = {
            (ep.scene_id, ep.episode_id) for ep in full_dataset.episodes
        }

        dataset_config.content_scenes = scenes[0 : len(scenes) // 2]
        split1_dataset = make_dataset(
            id_dataset=dataset_config.type, config=dataset_config
        )
        split1_episodes = {
            (ep.scene_id, ep.episode_id) for ep in split1_dataset.episodes
        }

        dataset_config.content_scenes = scenes[len(scenes) // 2 :]
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


def test_r2r_vln_sim():
    vln_config = get_config(CFG_TEST)

    if not r2r_vln_dataset.VLNDatasetV1.check_config_paths_exist(
        vln_config.habitat.dataset
    ):
        pytest.skip(
            "Please download Matterport3D R2R VLN dataset to data folder."
        )

    dataset = make_dataset(
        id_dataset=vln_config.habitat.dataset.type,
        config=vln_config.habitat.dataset,
    )

    with habitat.Env(config=vln_config, dataset=dataset) as env:
        env.episodes = dataset.episodes[:EPISODES_LIMIT]

        follower = ShortestPathFollower(
            env.sim, goal_radius=0.5, return_one_hot=False
        )

        for _ in range(len(env.episodes)):
            env.reset()
            path = env.current_episode.reference_path + [
                env.current_episode.goals[0].position
            ]
            for point in path:
                while env.episode_over:
                    best_action = follower.get_next_action(point)

                    obs = env.step(best_action)
                    assert "rgb" in obs, "RGB image is missing in observation."
                    assert (
                        "instruction" in obs
                    ), "Instruction is missing in observation."
                    assert (
                        obs["instruction"]["text"]
                        == env.current_episode.instruction.instruction_text
                    ), "Instruction from sensor does not match the instruction from the episode"
                    agent_config = get_agent_config(
                        vln_config.habitat.simulator
                    )
                    assert obs["rgb"].shape[:2] == (
                        agent_config.sim_sensors["rgb_sensor"].height,
                        agent_config.sim_sensors["rgb_sensor"].width,
                    ), (
                        "Observation resolution {} doesn't correspond to config "
                        "({}, {}).".format(
                            obs["rgb"].shape[:2],
                            vln_config.habitat.simulator["rgb_sensor"].height,
                            vln_config.habitat.simulator["rgb_sensor"].width,
                        )
                    )
