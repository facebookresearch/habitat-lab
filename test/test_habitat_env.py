#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing as mp
import os

import numpy as np
import pytest

import habitat
from habitat.config.default import get_config
from habitat.core.simulator import AgentState
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat.sims.habitat_simulator import (
    SimulatorActions,
    SIM_ACTION_TO_NAME,
    SIM_NAME_TO_ACTION,
)
from habitat.tasks.nav.nav_task import NavigationEpisode

CFG_TEST = "test/habitat_all_sensors_test.yaml"
NUM_ENVS = 2


class DummyRLEnv(habitat.RLEnv):
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


def _load_test_data():
    configs = []
    datasets = []
    for i in range(NUM_ENVS):
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


def _vec_env_test_fn(configs, datasets, multiprocessing_start_method):
    num_envs = len(configs)
    env_fn_args = tuple(zip(configs, datasets, range(num_envs)))
    envs = habitat.VectorEnv(
        env_fn_args=env_fn_args,
        multiprocessing_start_method=multiprocessing_start_method,
    )
    envs.reset()
    non_stop_actions = [
        k
        for k, v in SIM_ACTION_TO_NAME.items()
        if v != SimulatorActions.STOP.value
    ]

    for _ in range(2 * configs[0].ENVIRONMENT.MAX_EPISODE_STEPS):
        observations = envs.step(np.random.choice(non_stop_actions, num_envs))
        assert len(observations) == num_envs

    envs.close()


def test_vectorized_envs_forkserver():
    configs, datasets = _load_test_data()
    _vec_env_test_fn(configs, datasets, "forkserver")


def test_vectorized_envs_spawn():
    configs, datasets = _load_test_data()
    _vec_env_test_fn(configs, datasets, "spawn")


def _fork_test_target(configs, datasets):
    _vec_env_test_fn(configs, datasets, "fork")


def test_vectorized_envs_fork():
    configs, datasets = _load_test_data()
    # 'fork' works in a process that has yet to use the GPU
    # this test uses spawns a new python instance, which allows us to fork
    mp_ctx = mp.get_context("spawn")
    p = mp_ctx.Process(target=_fork_test_target, args=(configs, datasets))
    p.start()
    p.join()
    assert p.exitcode == 0


def test_threaded_vectorized_env():
    configs, datasets = _load_test_data()
    num_envs = len(configs)
    env_fn_args = tuple(zip(configs, datasets, range(num_envs)))
    envs = habitat.ThreadedVectorEnv(env_fn_args=env_fn_args)
    envs.reset()
    non_stop_actions = [
        k
        for k, v in SIM_ACTION_TO_NAME.items()
        if v != SimulatorActions.STOP.value
    ]

    for i in range(2 * configs[0].ENVIRONMENT.MAX_EPISODE_STEPS):
        observations = envs.step(np.random.choice(non_stop_actions, num_envs))
        assert len(observations) == num_envs

    envs.close()


def test_env():
    config = get_config(CFG_TEST)
    if not os.path.exists(config.SIMULATOR.SCENE):
        pytest.skip("Please download Habitat test data to data folder.")
    env = habitat.Env(config=config, dataset=None)
    env.episodes = [
        NavigationEpisode(
            episode_id="0",
            scene_id=config.SIMULATOR.SCENE,
            start_position=[03.00611, 0.072447, -2.67867],
            start_rotation=[0, 0.163276, 0, 0.98658],
            goals=[],
        )
    ]

    env.reset()
    non_stop_actions = [
        k
        for k, v in SIM_ACTION_TO_NAME.items()
        if v != SimulatorActions.STOP.value
    ]
    for _ in range(config.ENVIRONMENT.MAX_EPISODE_STEPS):
        env.step(np.random.choice(non_stop_actions))

    # check for steps limit on environment
    assert env.episode_over is True, (
        "episode should be over after " "max_episode_steps"
    )

    env.reset()

    env.step(SIM_NAME_TO_ACTION[SimulatorActions.STOP.value])
    # check for STOP action
    assert env.episode_over is True, (
        "episode should be over after STOP " "action"
    )

    env.close()


def make_rl_env(config, dataset, rank: int = 0):
    """Constructor for default habitat Env.
    :param config: configurations for environment
    :param dataset: dataset for environment
    :param rank: rank for setting seeds for environment
    :return: constructed habitat Env
    """
    env = DummyRLEnv(config=config, dataset=dataset)
    env.seed(config.SEED + rank)
    return env


def test_rl_vectorized_envs():
    configs, datasets = _load_test_data()

    num_envs = len(configs)
    env_fn_args = tuple(zip(configs, datasets, range(num_envs)))
    envs = habitat.VectorEnv(make_env_fn=make_rl_env, env_fn_args=env_fn_args)
    envs.reset()
    non_stop_actions = [
        k
        for k, v in SIM_ACTION_TO_NAME.items()
        if v != SimulatorActions.STOP.value
    ]

    for i in range(2 * configs[0].ENVIRONMENT.MAX_EPISODE_STEPS):
        outputs = envs.step(np.random.choice(non_stop_actions, num_envs))
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
        assert len(observations) == num_envs
        assert len(rewards) == num_envs
        assert len(dones) == num_envs
        assert len(infos) == num_envs
        if (i + 1) % configs[0].ENVIRONMENT.MAX_EPISODE_STEPS == 0:
            assert all(dones), "dones should be true after max_episode steps"

    envs.close()


def test_rl_env():
    config = get_config(CFG_TEST)
    if not os.path.exists(config.SIMULATOR.SCENE):
        pytest.skip("Please download Habitat test data to data folder.")

    env = DummyRLEnv(config=config, dataset=None)
    env.episodes = [
        NavigationEpisode(
            episode_id="0",
            scene_id=config.SIMULATOR.SCENE,
            start_position=[03.00611, 0.072447, -2.67867],
            start_rotation=[0, 0.163276, 0, 0.98658],
            goals=[],
        )
    ]

    done = False
    observation = env.reset()

    non_stop_actions = [
        k
        for k, v in SIM_ACTION_TO_NAME.items()
        if v != SimulatorActions.STOP.value
    ]
    for _ in range(config.ENVIRONMENT.MAX_EPISODE_STEPS):
        observation, reward, done, info = env.step(
            np.random.choice(non_stop_actions)
        )

    # check for steps limit on environment
    assert done is True, "episodes should be over after max_episode_steps"

    env.reset()
    observation, reward, done, info = env.step(
        SIM_NAME_TO_ACTION[SimulatorActions.STOP.value]
    )
    assert done is True, "done should be true after STOP action"

    env.close()


def test_action_space_shortest_path():
    config = get_config()
    if not os.path.exists(config.SIMULATOR.SCENE):
        pytest.skip("Please download Habitat test data to data folder.")

    env = habitat.Env(config=config, dataset=None)

    # action space shortest path
    source_position = env.sim.sample_navigable_point()
    angles = [x for x in range(-180, 180, config.SIMULATOR.TURN_ANGLE)]
    angle = np.radians(np.random.choice(angles))
    source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
    source = AgentState(source_position, source_rotation)

    reachable_targets = []
    unreachable_targets = []
    while len(reachable_targets) < 5:
        position = env.sim.sample_navigable_point()
        angles = [x for x in range(-180, 180, config.SIMULATOR.TURN_ANGLE)]
        angle = np.radians(np.random.choice(angles))
        rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
        if env.sim.geodesic_distance(source_position, position) != np.inf:
            reachable_targets.append(AgentState(position, rotation))

    while len(unreachable_targets) < 3:
        position = env.sim.sample_navigable_point()
        angles = [x for x in range(-180, 180, config.SIMULATOR.TURN_ANGLE)]
        angle = np.radians(np.random.choice(angles))
        rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
        if env.sim.geodesic_distance(source_position, position) == np.inf:
            unreachable_targets.append(AgentState(position, rotation))

    targets = reachable_targets
    shortest_path1 = env.sim.action_space_shortest_path(source, targets)
    assert shortest_path1 != []

    targets = unreachable_targets
    shortest_path2 = env.sim.action_space_shortest_path(source, targets)
    assert shortest_path2 == []

    targets = reachable_targets + unreachable_targets
    shortest_path3 = env.sim.action_space_shortest_path(source, targets)

    # shortest_path1 should be identical to shortest_path3
    assert len(shortest_path1) == len(shortest_path3)
    for i in range(len(shortest_path1)):
        assert np.allclose(
            shortest_path1[i].position, shortest_path3[i].position
        )
        assert np.allclose(
            shortest_path1[i].rotation, shortest_path3[i].rotation
        )
        assert shortest_path1[i].action == shortest_path3[i].action

    targets = unreachable_targets + [source]
    shortest_path4 = env.sim.action_space_shortest_path(source, targets)
    assert len(shortest_path4) == 1
    assert np.allclose(shortest_path4[0].position, source.position)
    assert np.allclose(shortest_path4[0].rotation, source.rotation)
    assert shortest_path4[0].action is None

    env.close()
