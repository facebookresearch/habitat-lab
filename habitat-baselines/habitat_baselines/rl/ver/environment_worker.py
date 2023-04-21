#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import random
from multiprocessing.context import BaseContext
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Sequence,
    TypeVar,
)

import attr
import numpy as np

from habitat import RLEnv, logger, make_dataset
from habitat.config import read_write
from habitat.gym import make_gym_from_config
from habitat.gym.gym_env_episode_count_wrapper import EnvCountEpisodeWrapper
from habitat.gym.gym_env_obs_dict_wrapper import EnvObsDictWrapper
from habitat_baselines.common.tensor_dict import NDArrayDict, TensorDict
from habitat_baselines.rl.ver.queue import BatchedQueue
from habitat_baselines.rl.ver.task_enums import (
    EnvironmentWorkerTasks,
    ReportWorkerTasks,
)
from habitat_baselines.rl.ver.worker_common import (
    ProcessBase,
    WorkerBase,
    WorkerQueues,
)
from habitat_baselines.utils.common import (
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.timing import Timing

if TYPE_CHECKING:
    from omegaconf import DictConfig


MIN_SCENES_PER_ENV = 16

T = TypeVar("T")


def infinite_shuffling_iterator(
    inp_seq: Sequence[T], prng: Optional[random.Random] = None
) -> Iterable[T]:
    inp = list(inp_seq)
    i = len(inp)
    while True:
        if i == len(inp):
            if prng is not None:
                prng.shuffle(inp)
            else:
                random.shuffle(inp)
            i = 0

        yield inp[i]
        i += 1


@attr.s(slots=True, init=False, auto_attribs=True)
class DefaultActionPlugin:
    policy_action_space: Any
    is_continuous: bool

    def __init__(self, policy_action_space: Any) -> None:
        self.policy_action_space = policy_action_space
        self.is_continuous = is_continuous_action_space(policy_action_space)

    def __call__(self, action: np.ndarray) -> np.ndarray:
        if self.is_continuous:
            return np.clip(
                action,
                self.policy_action_space.low,
                self.policy_action_space.high,
            )
        else:
            return action.item()


@attr.s(auto_attribs=True)
class EnvironmentWorkerProcess(ProcessBase):
    env_idx: int
    env_config: Any
    auto_reset_done: bool
    queues: WorkerQueues
    action_plugin: Callable[[np.ndarray], np.ndarray] = attr.ib(init=False)
    env: RLEnv = attr.ib(default=None)
    total_reward: float = 0.0
    episode_length: int = 0
    timer: Timing = attr.Factory(Timing)
    _episode_id: int = 0
    _step_id: int = 0
    _torch_transfer_buffers: TensorDict = attr.ib(init=False)
    send_transfer_buffers: NDArrayDict = attr.ib(init=False)
    actions: np.ndarray = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.build_dispatch_table(EnvironmentWorkerTasks)

    def wait(self):
        self.response_queue.put("REQ")

    def close(self):
        if self.env is not None:
            self.env.close()

        self.env = None

    def start(self):
        assert self.env is None
        self.env = EnvCountEpisodeWrapper(
            EnvObsDictWrapper(make_gym_from_config(self.env_config))
        )

        self.response_queue.put(
            dict(
                env_idx=self.env_idx,
                obs_space=self.env.observation_space,
                act_space=self.env.action_space,
            )
        )

        logger.info(f"EnvironmentWorker-{self.env_idx} initialized")

    def reset(self):
        self._last_obs = self.env.reset()

    def start_experience_collection(self):
        assert self.env is not None

        full_env_result = dict(
            observations=self._last_obs,
            episode_ids=self._episode_id,
            step_ids=self._step_id,
        )
        self.send_transfer_buffers.slice_keys(full_env_result.keys())[
            self.env_idx
        ] = full_env_result  # type:ignore[assignment]
        self.queues.inference.put(self.env_idx)

    def _step_env(self, action):
        with self.timer.avg_time("step env"):
            obs, reward, done, info = self.env.step(action)
            self._step_id += 1

            if not math.isfinite(reward):
                reward = -1.0

        with self.timer.avg_time("reset env"):
            if done:
                self._episode_id += 1
                self._step_id = 0
                if self.auto_reset_done:
                    obs = self.env.reset()  # type: ignore [assignment]

        return obs, reward, done, info

    def step(self):
        with self.timer.avg_time("process actions"):
            action = self.action_plugin(self.actions[self.env_idx])

        self._last_obs, reward, done, info = self._step_env(action)

        with self.timer.avg_time("enqueue env"):
            self.send_transfer_buffers[
                self.env_idx
            ] = dict(  # type:ignore[assignment]
                observations=self._last_obs,
                rewards=reward,
                masks=not done,
                episode_ids=self._episode_id,
                step_ids=self._step_id,
            )

        self.queues.inference.put(self.env_idx)

        self.total_reward += reward
        self.episode_length += 1

        if done:
            self.queues.report.put_many(
                [
                    (
                        ReportWorkerTasks.episode_end,
                        dict(
                            env_idx=self.env_idx,
                            length=self.episode_length,
                            reward=self.total_reward,
                            info=info,
                        ),
                    ),
                    (ReportWorkerTasks.env_timing, self.timer),
                ]
            )
            self.timer = Timing()
            self.total_reward = 0.0
            self.episode_length = 0

    @property
    def task_queue(self) -> BatchedQueue:
        return self.queues.environments[self.env_idx]

    def set_transfer_buffers(self, torch_transfer_buffers: TensorDict):
        self._torch_transfer_buffers = torch_transfer_buffers
        self._torch_transfer_buffers["environment_ids"][
            self.env_idx
        ] = self.env_idx  # type:ignore[assignment]
        acts = self._torch_transfer_buffers["actions"].numpy()
        assert isinstance(acts, np.ndarray)
        self.actions = acts
        self.send_transfer_buffers = self._torch_transfer_buffers.slice_keys(
            "observations",
            "rewards",
            "masks",
            "episode_ids",
            "step_ids",
        ).numpy()

    def set_action_plugin(self, action_plugin):
        self.action_plugin = action_plugin

    @inference_mode()
    def run(self):
        os.nice(10)

        super().run()

        self.close()


class EnvironmentWorker(WorkerBase):
    def __init__(
        self,
        mp_ctx: BaseContext,
        env_idx: int,
        env_config,
        auto_reset_done,
        queues: WorkerQueues,
    ):
        super().__init__(
            mp_ctx,
            EnvironmentWorkerProcess,
            env_idx,
            env_config,
            auto_reset_done,
            queues,
        )
        self.env_worker_queue = queues.environments[env_idx]

    def start(self):
        self.env_worker_queue.put((EnvironmentWorkerTasks.start, None))

    def get_init_report(self):
        return self.response_queue.get()

    def reset(self):
        self.env_worker_queue.put((EnvironmentWorkerTasks.reset, None))

    def set_transfer_buffers(self, buffers: TensorDict):
        self.env_worker_queue.put(
            (EnvironmentWorkerTasks.set_transfer_buffers, buffers)
        )

    def set_action_plugin(self, action_plugin):
        self.env_worker_queue.put(
            (EnvironmentWorkerTasks.set_action_plugin, action_plugin)
        )

    def start_experience_collection(self):
        self.env_worker_queue.put(
            (EnvironmentWorkerTasks.start_experience_collection, None)
        )

    def wait_start(self):
        self.env_worker_queue.put((EnvironmentWorkerTasks.wait, None))

    def wait_sync(self):
        assert (
            self.response_queue.get() == "REQ"
        ), "Got the wrong response from the actor worker. Something unexpected was placed in the response queue"

    def wait(self):
        self.wait_start()
        self.wait_sync()


def _construct_environment_workers_impl(
    configs,
    auto_reset_done,
    mp_ctx: BaseContext,
    queues: WorkerQueues,
):
    num_environments = len(configs)
    workers = []
    for i in range(num_environments):
        w = EnvironmentWorker(
            mp_ctx,
            i,
            configs[i],
            auto_reset_done,
            queues,
        )
        workers.append(w)

    return workers


def _make_proc_config(config, rank, scenes=None, scene_splits=None):
    proc_config = config.copy()
    with read_write(proc_config):
        task_config = proc_config.habitat
        task_config.seed = task_config.seed + rank
        if scenes is not None and len(scenes) > 0:
            task_config.dataset.content_scenes = scene_splits[rank]

    return proc_config


def _create_worker_configs(config: "DictConfig"):
    num_environments = config.habitat_baselines.num_environments

    dataset = make_dataset(config.habitat.dataset.type)
    scenes = config.habitat.dataset.content_scenes
    if "*" in config.habitat.dataset.content_scenes:
        scenes = dataset.get_scenes_to_load(config.habitat.dataset)

    # We use a minimum number of scenes per environment to reduce bias
    scenes_per_env = max(
        int(math.ceil(len(scenes) / num_environments)), MIN_SCENES_PER_ENV
    )
    scene_splits: List[List[str]] = [[] for _ in range(num_environments)]
    for idx, scene in enumerate(infinite_shuffling_iterator(scenes)):
        scene_splits[idx % len(scene_splits)].append(scene)
        if len(scene_splits[-1]) == scenes_per_env:
            break

    assert len(set().union(*(set(scenes) for scenes in scene_splits))) == len(
        scenes
    )

    args = [
        _make_proc_config(config, rank, scenes, scene_splits)
        for rank in range(num_environments)
    ]

    return args


def construct_environment_workers(
    config: "DictConfig",
    mp_ctx: BaseContext,
    worker_queues: WorkerQueues,
) -> List[EnvironmentWorker]:
    configs = _create_worker_configs(config)

    return _construct_environment_workers_impl(
        configs,
        True,
        mp_ctx,
        worker_queues,
    )


def build_action_plugin_from_policy_action_space(policy_action_space):
    return DefaultActionPlugin(policy_action_space)
