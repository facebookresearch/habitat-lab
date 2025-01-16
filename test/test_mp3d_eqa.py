#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

import numpy as np
import pytest

import habitat
from habitat.config.default import get_agent_config, get_config
from habitat.core.embodied_task import Episode
from habitat.core.logging import logger
from habitat.datasets import make_dataset
from habitat.datasets.eqa import mp3d_eqa_dataset as mp3d_dataset
from habitat.tasks.eqa.eqa import AnswerAction
from habitat.tasks.nav.nav import MoveForwardAction
from habitat.utils.geometry_utils import (
    angle_between_quaternions,
    quaternion_from_coeff,
)
from habitat.utils.test_utils import sample_non_stop_action

CFG_TEST = "test/config/habitat/habitat_mp3d_eqa_test.yaml"
CLOSE_STEP_THRESHOLD = 0.028
OLD_STOP_ACTION_ID = 3


# List of episodes each from unique house
TEST_EPISODE_SET = [1, 309, 807, 958, 696, 10, 297, 1021, 1307, 1569]

RGB_EPISODE_MEANS = {
    1: 123.20,
    10: 120.56,
    297: 122.69,
    309: 118.66,
    696: 116.10,
    807: 145.77,
    958: 143.48,
    1021: 119.10,
    1307: 102.11,
    1569: 91.01,
}

EPISODES_LIMIT = 6


def get_minos_for_sim_eqa_config():
    _sim_eqa_c = get_config(CFG_TEST)
    _sim_eqa_c.task_name = "EQA-v0"
    _sim_eqa_c.dataset = mp3d_dataset.get_default_mp3d_v1_config()
    _sim_eqa_c.dataset.split = "val"
    _sim_eqa_c.scene = "data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb"
    _sim_eqa_c.height = 512
    _sim_eqa_c.width = 512
    _sim_eqa_c.hfov = "45"
    _sim_eqa_c.vfov = "45"
    _sim_eqa_c.sensor_position = [0, 1.09, 0]
    _sim_eqa_c.forward_step_size = 0.1  # in metres
    _sim_eqa_c.turn_angle = 9  # in degrees
    _sim_eqa_c.sim = "Sim-v0"

    # Agent configuration
    agent_c = _sim_eqa_c.agents[0]
    agent_c.height = 1.5
    agent_c.radius = 0.1

    return _sim_eqa_c


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


def test_mp3d_eqa_dataset():
    dataset_config = get_config(CFG_TEST).habitat.dataset
    if not mp3d_dataset.Matterport3dDatasetV1.check_config_paths_exist(
        dataset_config
    ):
        pytest.skip("Please download Matterport3D EQA dataset to data folder.")

    dataset = mp3d_dataset.Matterport3dDatasetV1(config=dataset_config)
    assert dataset
    assert (
        len(dataset.episodes) == mp3d_dataset.EQA_MP3D_V1_VAL_EPISODE_COUNT
    ), "Test split episode number mismatch"
    check_json_serialization(dataset)


@pytest.mark.parametrize("split", ["train", "val"])
def test_dataset_splitting(split):
    dataset_config = get_config(CFG_TEST).habitat.dataset
    with habitat.config.read_write(dataset_config):
        dataset_config.split = split
        if not mp3d_dataset.Matterport3dDatasetV1.check_config_paths_exist(
            dataset_config
        ):
            pytest.skip(
                "Please download Matterport3D EQA dataset to data folder."
            )

        scenes = mp3d_dataset.Matterport3dDatasetV1.get_scenes_to_load(
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


def test_mp3d_eqa_sim():
    eqa_config = get_config(CFG_TEST)

    if not mp3d_dataset.Matterport3dDatasetV1.check_config_paths_exist(
        eqa_config.habitat.dataset
    ):
        pytest.skip("Please download Matterport3D EQA dataset to data folder.")

    dataset = make_dataset(
        id_dataset=eqa_config.habitat.dataset.type,
        config=eqa_config.habitat.dataset,
    )
    with habitat.Env(config=eqa_config, dataset=dataset) as env:
        env.episodes = dataset.episodes[:EPISODES_LIMIT]

        env.reset()
        while not env.episode_over:
            obs = env.step(env.task.action_space.sample())
            if not env.episode_over:
                assert "rgb" in obs, "RGB image is missing in observation."
                agent_config = get_agent_config(eqa_config.habitat.simulator)
                assert obs["rgb"].shape[:2] == (
                    agent_config.sim_sensors["rgb_sensor"].height,
                    agent_config.sim_sensors["rgb_sensor"].width,
                ), (
                    "Observation resolution {} doesn't correspond to config "
                    "({}, {}).".format(
                        obs["rgb"].shape[:2],
                        eqa_config.habitat.simulator["rgb_sensor"].height,
                        eqa_config.habitat.simulator["rgb_sensor"].width,
                    )
                )


def test_mp3d_eqa_sim_correspondence():
    eqa_config = get_config(CFG_TEST)

    if not mp3d_dataset.Matterport3dDatasetV1.check_config_paths_exist(
        eqa_config.habitat.dataset
    ):
        pytest.skip("Please download Matterport3D EQA dataset to data folder.")

    dataset = make_dataset(
        id_dataset=eqa_config.habitat.dataset.type,
        config=eqa_config.habitat.dataset,
    )
    with habitat.Env(config=eqa_config, dataset=dataset) as env:
        env.episodes = [
            episode
            for episode in dataset.episodes
            if int(episode.episode_id) in TEST_EPISODE_SET[:EPISODES_LIMIT]
        ]

        ep_i = 0
        cycles_n = 2
        while cycles_n > 0:
            env.reset()
            episode = env.current_episode
            assert (
                len(episode.goals) == 1
            ), "Episode has no goals or more than one."
            assert (
                len(episode.shortest_paths) == 1
            ), "Episode has no shortest paths or more than one."
            start_state = env.sim.get_agent_state()
            assert np.allclose(
                start_state.position, episode.start_position
            ), "Agent's start position diverges from the shortest path's one."

            rgb_mean = 0.0
            logger.info(
                "{id} {question}\n{answer}".format(
                    id=episode.episode_id,
                    question=episode.question.question_text,
                    answer=episode.question.answer_text,
                )
            )

            for step_id, point in enumerate(episode.shortest_paths[0]):
                cur_state = env.sim.get_agent_state()

                logger.info(
                    "diff position: {} diff rotation: {} \n"
                    "cur_state.position: {} shortest_path.position: {} \n"
                    "cur_state.rotation: {} shortest_path.rotation: {} action: {}\n"
                    "".format(
                        cur_state.position - point.position,
                        angle_between_quaternions(
                            cur_state.rotation,
                            quaternion_from_coeff(point.rotation),
                        ),
                        cur_state.position,
                        point.position,
                        cur_state.rotation,
                        point.rotation,
                        point.action,
                    )
                )

                assert np.allclose(
                    [cur_state.position[0], cur_state.position[2]],
                    [point.position[0], point.position[2]],
                    atol=CLOSE_STEP_THRESHOLD * (step_id + 1),
                ), "Agent's path diverges from the shortest path."

                if point.action != OLD_STOP_ACTION_ID:
                    obs = env.step(action=point.action)

                if not env.episode_over:
                    rgb_mean += obs["rgb"][:, :, :3].mean()

            if ep_i < len(RGB_EPISODE_MEANS):
                # Slightly bigger atol for basis meshes
                rgb_mean = rgb_mean / len(episode.shortest_paths[0])
                assert np.isclose(
                    RGB_EPISODE_MEANS[int(episode.episode_id)],
                    rgb_mean,
                    atol=0.5,
                ), f"RGB output doesn't match the ground truth. Expected {RGB_EPISODE_MEANS[int(episode.episode_id)]} but got {rgb_mean}"

            ep_i = (ep_i + 1) % EPISODES_LIMIT
            if ep_i == 0:
                cycles_n -= 1


def test_eqa_task():
    eqa_config = get_config(CFG_TEST)

    if not mp3d_dataset.Matterport3dDatasetV1.check_config_paths_exist(
        eqa_config.habitat.dataset
    ):
        pytest.skip("Please download Matterport3D EQA dataset to data folder.")

    dataset = make_dataset(
        id_dataset=eqa_config.habitat.dataset.type,
        config=eqa_config.habitat.dataset,
    )
    with habitat.Env(config=eqa_config, dataset=dataset) as env:
        env.episodes = list(
            filter(
                lambda e: int(e.episode_id)
                in TEST_EPISODE_SET[:EPISODES_LIMIT],
                dataset.episodes,
            )
        )

        env.reset()

        for _ in range(10):
            action = sample_non_stop_action(env.action_space)
            if action["action"] != AnswerAction.name:
                env.step(action)
            metrics = env.get_metrics()
            del metrics["episode_info"]
            logger.info(metrics)

        correct_answer_id = env.current_episode.question.answer_token
        env.step(
            {
                "action": AnswerAction.name,
                "action_args": {"answer_id": correct_answer_id},
            }
        )

        metrics = env.get_metrics()
        del metrics["episode_info"]
        logger.info(metrics)
        assert metrics["answer_accuracy"] == 1

        with pytest.raises(AssertionError):
            env.step({"action": MoveForwardAction.name})
