#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import multiprocessing as mp
import os
from typing import List

import numpy as np
import pytest
from gym import Wrapper

import habitat
from habitat.config.default import get_agent_config, get_config
from habitat.core.batch_rendering.env_batch_renderer_constants import (
    KEYFRAME_OBSERVATION_KEY,
)
from habitat.core.simulator import AgentState
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat.gym.gym_definitions import make_gym_from_config
from habitat.gym.gym_wrapper import HabGymWrapper
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.utils.test_utils import (
    sample_non_stop_action,
    sample_non_stop_action_gym,
)

CFG_TEST = "test/config/habitat/habitat_all_sensors_test.yaml"
NUM_ENVS = 4


class DummyRLEnv(habitat.RLEnv):
    def __init__(self, config, dataset=None):
        super(DummyRLEnv, self).__init__(config, dataset)

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


class CallTestEnvWrapper(Wrapper):
    def __init__(self, env, env_ind=0):
        super(CallTestEnvWrapper, self).__init__(env)
        self._dummy_variable = env_ind

    def get_env_ind(self):
        return self._dummy_variable

    def set_env_ind(self, new_env_ind):
        self._dummy_variable = new_env_ind


def _load_test_data():
    configs = []
    datasets = []
    for _ in range(NUM_ENVS):
        config = get_config(CFG_TEST)
        if not PointNavDatasetV1.check_config_paths_exist(
            config.habitat.dataset
        ):
            pytest.skip("Please download Habitat test data to data folder.")

        datasets.append(
            habitat.make_dataset(
                id_dataset=config.habitat.dataset.type,
                config=config.habitat.dataset,
            )
        )

        with habitat.config.read_write(config):
            config.habitat.simulator.scene = datasets[-1].episodes[0].scene_id
            # remove the teleport action that makes the action space continuous
            if "teleport" in config.habitat.task.actions:
                del config.habitat.task.actions["teleport"]
            if not os.path.exists(config.habitat.simulator.scene):
                pytest.skip(
                    "Please download Habitat test data to data folder."
                )
        configs.append(config)

    return configs, datasets


def _vec_env_test_fn(configs, datasets, multiprocessing_start_method, gpu2gpu):
    num_envs = len(configs)
    for cfg in configs:
        with habitat.config.read_write(cfg):
            cfg.habitat.simulator.habitat_sim_v0.gpu_gpu = gpu2gpu

    env_fn_args = tuple(zip(configs, datasets, range(num_envs)))
    with habitat.VectorEnv(
        make_env_fn=_make_dummy_env_func,
        env_fn_args=env_fn_args,
        multiprocessing_start_method=multiprocessing_start_method,
    ) as envs:
        envs.reset()

        for _ in range(2 * configs[0].habitat.environment.max_episode_steps):
            observations = envs.step(
                sample_non_stop_action_gym(envs.action_spaces[0], num_envs)
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
    configs, _ = _load_test_data()
    env_fn_args = tuple((c,) for c in configs)
    with habitat.VectorEnv(
        make_env_fn=make_gym_from_config,
        env_fn_args=env_fn_args,
        multiprocessing_start_method="forkserver",
    ) as envs:
        envs.reset()

    assert envs._is_closed


def test_number_of_episodes():
    configs, _ = _load_test_data()
    env_fn_args = tuple((c,) for c in configs)
    with habitat.VectorEnv(
        make_env_fn=make_gym_from_config,
        env_fn_args=env_fn_args,
        multiprocessing_start_method="forkserver",
    ) as envs:
        assert envs.number_of_episodes == [10000, 10000, 10000, 10000]


def test_threaded_vectorized_env():
    configs, datasets = _load_test_data()
    num_envs = len(configs)
    env_fn_args = tuple((c,) for c in configs)
    with habitat.ThreadedVectorEnv(
        make_env_fn=make_gym_from_config, env_fn_args=env_fn_args
    ) as envs:
        envs.reset()

        for _ in range(2 * configs[0].habitat.environment.max_episode_steps):
            observations = envs.step(
                sample_non_stop_action_gym(envs.action_spaces[0], num_envs)
            )
            assert len(observations) == num_envs


@pytest.mark.parametrize("gpu2gpu", [False, True])
def test_env(gpu2gpu):
    import habitat_sim

    if gpu2gpu and not habitat_sim.cuda_enabled:
        pytest.skip("GPU-GPU requires CUDA")

    config = get_config(CFG_TEST)
    if not os.path.exists(config.habitat.simulator.scene):
        pytest.skip("Please download Habitat test data to data folder.")

    with habitat.config.read_write(config):
        config.habitat.simulator.habitat_sim_v0.gpu_gpu = gpu2gpu
        # remove the teleport action that makes the action space continuous
        if "teleport" in config.habitat.task.actions:
            del config.habitat.task.actions["teleport"]
    with habitat.Env(config=config, dataset=None) as env:
        env.episodes = [
            NavigationEpisode(
                episode_id="0",
                scene_id=config.habitat.simulator.scene,
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

        for _ in range(config.habitat.environment.max_episode_steps):
            env.step(sample_non_stop_action(env.action_space))

        # check for steps limit on environment
        assert env.episode_over is True, (
            "episode should be over after " "max_episode_steps"
        )

        env.reset()

        env.step(action=0)
        # check for stop action
        assert (
            env.episode_over is True
        ), "episode should be over after stop action"


@pytest.mark.parametrize("gpu2gpu", [False, True])
def test_rl_vectorized_envs(gpu2gpu):
    import habitat_sim

    if gpu2gpu and not habitat_sim.cuda_enabled:
        pytest.skip("GPU-GPU requires CUDA")

    configs, datasets = _load_test_data()
    for config in configs:
        with habitat.config.read_write(config):
            config.habitat.simulator.habitat_sim_v0.gpu_gpu = gpu2gpu
            agent_config = get_agent_config(config.habitat.simulator)
            # Only keep the rgb_sensor
            agent_config.sim_sensors = {
                "rgb_sensor": agent_config.sim_sensors["rgb_sensor"]
            }

    num_envs = len(configs)
    env_fn_args = tuple(zip(configs, datasets, range(num_envs)))
    with habitat.VectorEnv(
        make_env_fn=_make_dummy_env_func, env_fn_args=env_fn_args
    ) as envs:
        envs.reset()

        for i in range(2 * configs[0].habitat.environment.max_episode_steps):
            outputs = envs.step(
                sample_non_stop_action_gym(envs.action_spaces[0], num_envs)
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
            h, w, c = observations[0]["rgb"].shape
            assert tiled_img.shape == (
                h * new_height,
                w * new_width,
                c,
            ), "vector env render is broken"

            if (i + 1) % configs[0].habitat.environment.max_episode_steps == 0:
                assert all(
                    dones
                ), "dones should be true after max_episode steps"


@pytest.mark.parametrize("classic_replay_renderer", [False, True])
@pytest.mark.parametrize("sensor_uuid", ["rgb_sensor", "depth_sensor"])
@pytest.mark.parametrize("gpu2gpu", [False])
def test_rl_vectorized_envs_batch_renderer(
    gpu2gpu: bool, sensor_uuid: str, classic_replay_renderer: bool
):
    import habitat_sim

    if gpu2gpu and not habitat_sim.cuda_enabled:
        pytest.skip("GPU-GPU requires CUDA")

    if sensor_uuid == "rgb_sensor":
        obs_key = "rgb"
    elif sensor_uuid == "depth_sensor":
        obs_key = "depth"
    else:
        pytest.fail("Unknown sensor uuid: " + sensor_uuid)

    configs, datasets = _load_test_data()
    for config in configs:
        with habitat.config.read_write(config):
            config.habitat.simulator.renderer.enable_batch_renderer = True
            config.habitat.simulator.renderer.classic_replay_renderer = (
                classic_replay_renderer
            )
            config.habitat.simulator.habitat_sim_v0.enable_gfx_replay_save = (
                True
            )
            config.habitat.simulator.create_renderer = False
            config.habitat.simulator.habitat_sim_v0.gpu_gpu = gpu2gpu
            agent_config = get_agent_config(config.habitat.simulator)
            ## Only keep one sensor
            agent_config.sim_sensors = {
                sensor_uuid: agent_config.sim_sensors[sensor_uuid]
            }

    num_envs = len(configs)
    env_fn_args = tuple(zip(configs, datasets, range(num_envs)))
    with habitat.VectorEnv(
        make_env_fn=_make_dummy_env_func, env_fn_args=env_fn_args
    ) as envs:
        envs.initialize_batch_renderer(configs[0])
        observations = envs.reset()
        for env_obs in observations:
            assert KEYFRAME_OBSERVATION_KEY in env_obs

        observations = envs.post_step(observations)
        for env_obs in observations:
            assert KEYFRAME_OBSERVATION_KEY not in env_obs

        assert len(observations) == num_envs

        # TODO: Add screenshot tests. Image stats are compared until then.
        threshold: float = 0.01
        if sensor_uuid == "rgb_sensor":
            baseline_mean = [126.23, 126.84, 126.62, 125.63]
            baseline_std_dev = [26.54, 26.16, 25.80, 26.56]
        elif sensor_uuid == "depth_sensor":
            baseline_mean = [0.4852, 0.4886, 0.4920, 0.4956]
            baseline_std_dev = [0.0107, 0.0112, 0.0117, 0.0122]
        for env_idx in range(num_envs):
            env_obs = observations[env_idx][obs_key]
            mean = float(np.mean(env_obs[env_idx]))
            std_dev = float(np.std(env_obs[env_idx]))
            assert abs(baseline_mean[env_idx] - mean) < threshold
            assert abs(baseline_std_dev[env_idx] - std_dev) < threshold

        for _ in range(2 * configs[0].habitat.environment.max_episode_steps):
            outputs = envs.step(
                sample_non_stop_action_gym(envs.action_spaces[0], num_envs)
            )
            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]

            for env_obs in observations:
                assert KEYFRAME_OBSERVATION_KEY in env_obs

            observations = envs.post_step(observations)

            for env_obs in observations:
                assert KEYFRAME_OBSERVATION_KEY not in env_obs

            assert len(observations) == num_envs
            assert len(rewards) == num_envs
            assert len(dones) == num_envs
            assert len(infos) == num_envs

            # Note: Uncomment this line to visualize:
            # envs.render(mode="human")

            tiled_img = envs.render(mode="rgb_array")
            new_height = int(np.ceil(np.sqrt(NUM_ENVS)))
            new_width = int(np.ceil(float(NUM_ENVS) / new_height))
            h, w, _ = observations[0][obs_key].shape
            assert tiled_img.shape == (
                h * new_height,
                w * new_width,
                3,
            ), "vector env render is broken"


@pytest.mark.parametrize("gpu2gpu", [False, True])
def test_rl_env(gpu2gpu):
    import habitat_sim

    if gpu2gpu and not habitat_sim.cuda_enabled:
        pytest.skip("GPU-GPU requires CUDA")

    config = get_config(CFG_TEST)
    if not os.path.exists(config.habitat.simulator.scene):
        pytest.skip("Please download Habitat test data to data folder.")

    with habitat.config.read_write(config):
        config.habitat.simulator.habitat_sim_v0.gpu_gpu = gpu2gpu
        # remove the teleport action that makes the action space continuous
        if "teleport" in config.habitat.task.actions:
            del config.habitat.task.actions["teleport"]

    with _make_dummy_env_func(config=config, dataset=None) as env:
        env.episodes = [
            NavigationEpisode(
                episode_id="0",
                scene_id=config.habitat.simulator.scene,
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

        for _ in range(config.habitat.environment.max_episode_steps):
            observation, reward, done, info = env.step(
                action=sample_non_stop_action_gym(env.action_space)
            )

        # check for steps limit on environment
        assert done is True, "episodes should be over after max_episode_steps"

        env.reset()
        observation, reward, done, info = env.step(action=0)
        assert done is True, "done should be true after stop action"


def _make_dummy_env_func(config, dataset=None, env_id=0, rank=0):
    r"""Constructor for dummy habitat Env.
    :param config: configurations for environment
    :param dataset: dataset for environment
    :param rank: rank for setting seeds for environment
    :return: constructed habitat Env
    """
    env = DummyRLEnv(config=config, dataset=dataset)
    env.seed(config.habitat.seed + rank)
    env = HabGymWrapper(env=env)
    env = CallTestEnvWrapper(env, env_id)
    return env


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
    configs, _ = _load_test_data()
    env_fn_args = tuple((c,) for c in configs)
    with habitat.VectorEnv(
        make_env_fn=make_gym_from_config,
        env_fn_args=env_fn_args,
        multiprocessing_start_method="forkserver",
    ) as envs:
        envs.reset()

        envs.pause_at(3)
        envs.pause_at(0)

    assert envs._is_closed


# TODO Bring back this test for the greedy follower
@pytest.mark.skip
def test_action_space_shortest_path():
    config = get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml")
    if not os.path.exists(config.habitat.simulator.scene):
        pytest.skip("Please download Habitat test data to data folder.")

    env = habitat.Env(config=config, dataset=None)

    # action space shortest path
    source_position = env.sim.sample_navigable_point()
    angles = list(range(-180, 180, config.habitat.simulator.turn_angle))
    angle = np.radians(np.random.choice(angles))
    source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
    source = AgentState(source_position, source_rotation)

    reachable_targets: List[AgentState] = []
    unreachable_targets: List[AgentState] = []
    while len(reachable_targets) < 5:
        position = env.sim.sample_navigable_point()
        angles = list(range(-180, 180, config.habitat.simulator.turn_angle))
        angle = np.radians(np.random.choice(angles))
        rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
        if env.sim.geodesic_distance(source_position, [position]) != np.inf:
            reachable_targets.append(AgentState(position, rotation))

    while len(unreachable_targets) < 3:
        position = env.sim.sample_navigable_point()
        # Change height of the point to make it unreachable
        position[1] = 100
        angles = list(range(-180, 180, config.habitat.simulator.turn_angle))
        angle = np.radians(np.random.choice(angles))
        rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
        if env.sim.geodesic_distance(source_position, [position]) == np.inf:
            unreachable_targets.append(AgentState(position, rotation))

    targets = reachable_targets
    shortest_path1 = env.action_space_shortest_path(  # type: ignore[attr-defined]
        source, targets
    )
    assert shortest_path1 != []

    targets = unreachable_targets
    shortest_path2 = env.action_space_shortest_path(  # type: ignore[attr-defined]
        source, targets
    )
    assert shortest_path2 == []
    env.close()


@pytest.mark.parametrize("set_method", ["current", "list", "iter"])
def test_set_episodes(set_method):
    config = get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml")
    if not os.path.exists(config.habitat.simulator.scene):
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
