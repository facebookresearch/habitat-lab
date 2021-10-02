#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import multiprocessing as mp
import os

import numpy as np
import pytest

import habitat
from habitat.config.default import get_config
from habitat.core.simulator import AgentState
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal, StopAction
from habitat.utils.test_utils import sample_non_stop_action

CFG_TEST = "configs/test/habitat_all_sensors_test.yaml"
NUM_ENVS = 4


class DummyRLEnv(habitat.RLEnv):
    def __init__(self, config, dataset=None, env_ind=0):
        super(DummyRLEnv, self).__init__(config, dataset)
        self._env_ind = env_ind

    def get_reward_range(self):
        return -1.0, 1.0

    def get_reward(self, observations):
        return 0.0

    def get_done(self, observations):
        done = False
        if self._env.episode_over:
            done = True
        return done

    def get_info(self, observations):
        return {}

    def get_env_ind(self):
        return self._env_ind

    def set_env_ind(self, new_env_ind):
        self._env_ind = new_env_ind


def _load_test_data():
    configs = []
    datasets = []
    for _ in range(NUM_ENVS):
        config = get_config(CFG_TEST)
        if not PointNavDatasetV1.check_config_paths_exist(config.DATASET):
            pytest.skip("Please download Habitat test data to data folder.")

        datasets.append(
            habitat.make_dataset(
                id_dataset=config.DATASET.TYPE, config=config.DATASET
            )
        )

        config.defrost()
        config.SIMULATOR.SCENE = datasets[-1].episodes[0].scene_id
        if not os.path.exists(config.SIMULATOR.SCENE):
            pytest.skip("Please download Habitat test data to data folder.")
        config.freeze()
        configs.append(config)

    return configs, datasets


def _vec_env_test_fn(configs, datasets, multiprocessing_start_method, gpu2gpu):
    num_envs = len(configs)
    for cfg in configs:
        cfg.defrost()
        cfg.SIMULATOR.HABITAT_SIM_V0.GPU_GPU = gpu2gpu
        cfg.freeze()

    env_fn_args = tuple(zip(configs, datasets, range(num_envs)))
    with habitat.VectorEnv(
        env_fn_args=env_fn_args,
        multiprocessing_start_method=multiprocessing_start_method,
    ) as envs:
        envs.reset()

        for _ in range(2 * configs[0].ENVIRONMENT.MAX_EPISODE_STEPS):
            observations = envs.step(
                sample_non_stop_action(envs.action_spaces[0], num_envs)
            )
            assert len(observations) == num_envs


@pytest.mark.parametrize(
    "multiprocessing_start_method,gpu2gpu",
    itertools.product(["forkserver", "spawn", "fork"], [True, False]),
)
def test_vectorized_envs(multiprocessing_start_method, gpu2gpu):
    import habitat_sim

    if gpu2gpu and not habitat_sim.cuda_enabled:
        pytest.skip("GPU-GPU requires CUDA")

    configs, datasets = _load_test_data()
    if multiprocessing_start_method == "fork":
        if gpu2gpu:
            pytest.skip("Fork does not support gpu2gpu")

        # 'fork' works in a process that has yet to use the GPU
        # this test uses spawns a new python instance, which allows us to fork
        mp_ctx = mp.get_context("spawn")
        p = mp_ctx.Process(
            target=_vec_env_test_fn,
            args=(configs, datasets, multiprocessing_start_method, gpu2gpu),
        )
        p.start()
        p.join()
        assert p.exitcode == 0
    else:
        _vec_env_test_fn(
            configs, datasets, multiprocessing_start_method, gpu2gpu
        )


def test_with_scope():
    configs, datasets = _load_test_data()
    num_envs = len(configs)
    env_fn_args = tuple(zip(configs, datasets, range(num_envs)))
    with habitat.VectorEnv(
        env_fn_args=env_fn_args, multiprocessing_start_method="forkserver"
    ) as envs:
        envs.reset()

    assert envs._is_closed


def test_number_of_episodes():
    configs, datasets = _load_test_data()
    num_envs = len(configs)
    env_fn_args = tuple(zip(configs, datasets, range(num_envs)))
    with habitat.VectorEnv(
        env_fn_args=env_fn_args, multiprocessing_start_method="forkserver"
    ) as envs:
        assert envs.number_of_episodes == [10000, 10000, 10000, 10000]


def test_threaded_vectorized_env():
    configs, datasets = _load_test_data()
    num_envs = len(configs)
    env_fn_args = tuple(zip(configs, datasets, range(num_envs)))
    with habitat.ThreadedVectorEnv(env_fn_args=env_fn_args) as envs:
        envs.reset()

        for _ in range(2 * configs[0].ENVIRONMENT.MAX_EPISODE_STEPS):
            observations = envs.step(
                sample_non_stop_action(envs.action_spaces[0], num_envs)
            )
            assert len(observations) == num_envs


@pytest.mark.parametrize("gpu2gpu", [False, True])
def test_env(gpu2gpu):
    import habitat_sim

    if gpu2gpu and not habitat_sim.cuda_enabled:
        pytest.skip("GPU-GPU requires CUDA")

    config = get_config(CFG_TEST)
    if not os.path.exists(config.SIMULATOR.SCENE):
        pytest.skip("Please download Habitat test data to data folder.")

    config.defrost()
    config.SIMULATOR.HABITAT_SIM_V0.GPU_GPU = gpu2gpu
    config.freeze()
    with habitat.Env(config=config, dataset=None) as env:
        env.episodes = [
            NavigationEpisode(
                episode_id="0",
                scene_id=config.SIMULATOR.SCENE,
                start_position=[-3.0133917, 0.04623024, 7.3064547],
                start_rotation=[0, 0.163276, 0, 0.98658],
                goals=[
                    NavigationGoal(
                        position=[
                            -3.0133917 + 0.01,
                            0.04623024,
                            7.3064547 + 0.01,
                        ]
                    )
                ],
                info={"geodesic_distance": 0.001},
            )
        ]
        env.reset()

        for _ in range(config.ENVIRONMENT.MAX_EPISODE_STEPS):
            env.step(sample_non_stop_action(env.action_space))

        # check for steps limit on environment
        assert env.episode_over is True, (
            "episode should be over after " "max_episode_steps"
        )

        env.reset()

        env.step(action={"action": StopAction.name})
        # check for STOP action
        assert (
            env.episode_over is True
        ), "episode should be over after STOP action"


def make_rl_env(config, dataset, rank: int = 0):
    r"""Constructor for default habitat Env.
    :param config: configurations for environment
    :param dataset: dataset for environment
    :param rank: rank for setting seeds for environment
    :return: constructed habitat Env
    """
    env = DummyRLEnv(config=config, dataset=dataset)
    env.seed(config.SEED + rank)
    return env


@pytest.mark.parametrize("gpu2gpu", [False, True])
def test_rl_vectorized_envs(gpu2gpu):
    import habitat_sim

    if gpu2gpu and not habitat_sim.cuda_enabled:
        pytest.skip("GPU-GPU requires CUDA")

    configs, datasets = _load_test_data()
    for config in configs:
        config.defrost()
        config.SIMULATOR.HABITAT_SIM_V0.GPU_GPU = gpu2gpu
        config.freeze()

    num_envs = len(configs)
    env_fn_args = tuple(zip(configs, datasets, range(num_envs)))
    with habitat.VectorEnv(
        make_env_fn=make_rl_env, env_fn_args=env_fn_args
    ) as envs:
        envs.reset()

        for i in range(2 * configs[0].ENVIRONMENT.MAX_EPISODE_STEPS):
            outputs = envs.step(
                sample_non_stop_action(envs.action_spaces[0], num_envs)
            )
            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            assert len(observations) == num_envs
            assert len(rewards) == num_envs
            assert len(dones) == num_envs
            assert len(infos) == num_envs

            tiled_img = envs.render(mode="rgb_array")
            new_height = int(np.ceil(np.sqrt(NUM_ENVS)))
            new_width = int(np.ceil(float(NUM_ENVS) / new_height))
            print(f"observations: {observations}")
            h, w, c = observations[0]["rgb"].shape
            assert tiled_img.shape == (
                h * new_height,
                w * new_width,
                c,
            ), "vector env render is broken"

            if (i + 1) % configs[0].ENVIRONMENT.MAX_EPISODE_STEPS == 0:
                assert all(
                    dones
                ), "dones should be true after max_episode steps"


@pytest.mark.parametrize("gpu2gpu", [False, True])
def test_rl_env(gpu2gpu):
    import habitat_sim

    if gpu2gpu and not habitat_sim.cuda_enabled:
        pytest.skip("GPU-GPU requires CUDA")

    config = get_config(CFG_TEST)
    if not os.path.exists(config.SIMULATOR.SCENE):
        pytest.skip("Please download Habitat test data to data folder.")

    config.defrost()
    config.SIMULATOR.HABITAT_SIM_V0.GPU_GPU = gpu2gpu
    config.freeze()

    with DummyRLEnv(config=config, dataset=None) as env:
        env.episodes = [
            NavigationEpisode(
                episode_id="0",
                scene_id=config.SIMULATOR.SCENE,
                start_position=[-3.0133917, 0.04623024, 7.3064547],
                start_rotation=[0, 0.163276, 0, 0.98658],
                goals=[
                    NavigationGoal(
                        position=[
                            -3.0133917 + 0.01,
                            0.04623024,
                            7.3064547 + 0.01,
                        ]
                    )
                ],
                info={"geodesic_distance": 0.001},
            )
        ]

        done = False
        env.reset()

        for _ in range(config.ENVIRONMENT.MAX_EPISODE_STEPS):
            observation, reward, done, info = env.step(
                action=sample_non_stop_action(env.action_space)
            )

        # check for steps limit on environment
        assert done is True, "episodes should be over after max_episode_steps"

        env.reset()
        observation, reward, done, info = env.step(
            action={"action": StopAction.name}
        )
        assert done is True, "done should be true after STOP action"


def _make_dummy_env_func(config, dataset, env_id):
    return DummyRLEnv(config=config, dataset=dataset, env_ind=env_id)


def test_vec_env_call_func():
    configs, datasets = _load_test_data()
    num_envs = len(configs)
    env_fn_args = tuple(zip(configs, datasets, range(num_envs)))
    true_env_ids = list(range(num_envs))
    with habitat.VectorEnv(
        make_env_fn=_make_dummy_env_func,
        env_fn_args=env_fn_args,
        multiprocessing_start_method="forkserver",
    ) as envs:
        envs.reset()
        env_ids = envs.call(["get_env_ind"] * num_envs)
        assert env_ids == true_env_ids

        env_id = envs.call_at(1, "get_env_ind")
        assert env_id == true_env_ids[1]

        envs.call_at(2, "set_env_ind", {"new_env_ind": 20})
        true_env_ids[2] = 20
        env_ids = envs.call(["get_env_ind"] * num_envs)
        assert env_ids == true_env_ids

        envs.call_at(2, "set_env_ind", {"new_env_ind": 2})
        true_env_ids[2] = 2
        env_ids = envs.call(["get_env_ind"] * num_envs)
        assert env_ids == true_env_ids

        envs.pause_at(0)
        true_env_ids.pop(0)
        env_ids = envs.call(["get_env_ind"] * num_envs)
        assert env_ids == true_env_ids

        envs.pause_at(0)
        true_env_ids.pop(0)
        env_ids = envs.call(["get_env_ind"] * num_envs)
        assert env_ids == true_env_ids

        envs.resume_all()
        env_ids = envs.call(["get_env_ind"] * num_envs)
        assert env_ids == list(range(num_envs))


def test_close_with_paused():
    configs, datasets = _load_test_data()
    num_envs = len(configs)
    env_fn_args = tuple(zip(configs, datasets, range(num_envs)))
    with habitat.VectorEnv(
        env_fn_args=env_fn_args, multiprocessing_start_method="forkserver"
    ) as envs:
        envs.reset()

        envs.pause_at(3)
        envs.pause_at(0)

    assert envs._is_closed


# TODO Bring back this test for the greedy follower
@pytest.mark.skip
def test_action_space_shortest_path():
    config = get_config()
    if not os.path.exists(config.SIMULATOR.SCENE):
        pytest.skip("Please download Habitat test data to data folder.")

    env = habitat.Env(config=config, dataset=None)

    # action space shortest path
    source_position = env.sim.sample_navigable_point()
    angles = list(range(-180, 180, config.SIMULATOR.TURN_ANGLE))
    angle = np.radians(np.random.choice(angles))
    source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
    source = AgentState(source_position, source_rotation)

    reachable_targets = []
    unreachable_targets = []
    while len(reachable_targets) < 5:
        position = env.sim.sample_navigable_point()
        angles = list(range(-180, 180, config.SIMULATOR.TURN_ANGLE))
        angle = np.radians(np.random.choice(angles))
        rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
        if env.sim.geodesic_distance(source_position, [position]) != np.inf:
            reachable_targets.append(AgentState(position, rotation))

    while len(unreachable_targets) < 3:
        position = env.sim.sample_navigable_point()
        # Change height of the point to make it unreachable
        position[1] = 100
        angles = list(range(-180, 180, config.SIMULATOR.TURN_ANGLE))
        angle = np.radians(np.random.choice(angles))
        rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
        if env.sim.geodesic_distance(source_position, [position]) == np.inf:
            unreachable_targets.append(AgentState(position, rotation))

    targets = reachable_targets
    shortest_path1 = env.action_space_shortest_path(source, targets)
    assert shortest_path1 != []

    targets = unreachable_targets
    shortest_path2 = env.action_space_shortest_path(source, targets)
    assert shortest_path2 == []
    env.close()


@pytest.mark.parametrize("set_method", ["current", "list", "iter"])
def test_set_episodes(set_method):
    config = get_config()
    if not os.path.exists(config.SIMULATOR.SCENE):
        pytest.skip("Please download Habitat test data to data folder.")

    with habitat.Env(config=config, dataset=None) as env:
        all_episodes = list(env.episodes)
        target_episode = all_episodes[10]

        if set_method == "current":
            env.current_episode = target_episode
        elif set_method == "list":
            env.episodes = [target_episode]
        elif set_method == "iter":
            env.episode_iterator = iter([target_episode] + all_episodes)
        else:
            raise RuntimeError(
                f"Test does not support setting episodes with {set_method}"
            )

        env.reset()
        assert env.current_episode is target_episode
