#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import enum
import random
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, cast

import gym
import numba
import numpy as np
from gym import spaces

from habitat.config import Config
from habitat.core.dataset import Dataset, Episode, EpisodeIterator
from habitat.core.embodied_task import EmbodiedTask, Metrics
from habitat.core.logging import logger
from habitat.core.simulator import Observations, Simulator
from habitat.datasets import make_dataset
from habitat.sims import make_sim
from habitat.tasks import make_task
from habitat.utils import profiling_wrapper


class Env:
    r"""Fundamental environment class for :ref:`habitat`.

    :data observation_space: ``SpaceDict`` object corresponding to sensor in
        sim and task.
    :data action_space: ``gym.space`` object corresponding to valid actions.

    All the information  needed for working on embodied tasks with simulator
    is abstracted inside :ref:`Env`. Acts as a base for other derived
    environment classes. :ref:`Env` consists of three major components:
    ``dataset`` (`episodes`), ``simulator`` (:ref:`sim`) and :ref:`task` and
    connects all the three components together.
    """

    observation_space: spaces.Dict
    action_space: spaces.Dict
    _config: Config
    _dataset: Optional[Dataset]
    number_of_episodes: Optional[int]
    _episodes: List[Episode]
    _current_episode_index: Optional[int]
    _current_episode: Optional[Episode]
    _episode_iterator: Optional[Iterator]
    _sim: Simulator
    _task: EmbodiedTask
    _max_episode_seconds: int
    _max_episode_steps: int
    _elapsed_steps: int
    _episode_start_time: Optional[float]
    _episode_over: bool

    def __init__(
        self, config: Config, dataset: Optional[Dataset] = None
    ) -> None:
        """Constructor

        :param config: config for the environment. Should contain id for
            simulator and ``task_name`` which are passed into ``make_sim`` and
            ``make_task``.
        :param dataset: reference to dataset for task instance level
            information. Can be defined as :py:`None` in which case
            ``_episodes`` should be populated from outside.
        """

        assert config.is_frozen(), (
            "Freeze the config before creating the "
            "environment, use config.freeze()."
        )
        self._config = config
        self._dataset = dataset
        self._current_episode_index = None
        if self._dataset is None and config.DATASET.TYPE:
            self._dataset = make_dataset(
                id_dataset=config.DATASET.TYPE, config=config.DATASET
            )
        self._episodes = (
            self._dataset.episodes
            if self._dataset
            else cast(List[Episode], [])
        )
        self._current_episode = None
        iter_option_dict = {
            k.lower(): v
            for k, v in config.ENVIRONMENT.ITERATOR_OPTIONS.items()
        }
        iter_option_dict["seed"] = config.SEED
        self._episode_iterator = self._dataset.get_episode_iterator(
            **iter_option_dict
        )

        # load the first scene if dataset is present
        if self._dataset:
            assert (
                len(self._dataset.episodes) > 0
            ), "dataset should have non-empty episodes list"
            self._config.defrost()
            self._config.SIMULATOR.SCENE = self._dataset.episodes[0].scene_id
            self._config.freeze()

            self.number_of_episodes = len(self._dataset.episodes)
        else:
            self.number_of_episodes = None

        self._sim = make_sim(
            id_sim=self._config.SIMULATOR.TYPE, config=self._config.SIMULATOR
        )
        self._task = make_task(
            self._config.TASK.TYPE,
            config=self._config.TASK,
            sim=self._sim,
            dataset=self._dataset,
        )
        self.observation_space = spaces.Dict(
            {
                **self._sim.sensor_suite.observation_spaces.spaces,
                **self._task.sensor_suite.observation_spaces.spaces,
            }
        )
        self.action_space = self._task.action_space
        self._max_episode_seconds = (
            self._config.ENVIRONMENT.MAX_EPISODE_SECONDS
        )
        self._max_episode_steps = self._config.ENVIRONMENT.MAX_EPISODE_STEPS
        self._elapsed_steps = 0
        self._episode_start_time: Optional[float] = None
        self._episode_over = False

    @property
    def current_episode(self) -> Episode:
        assert self._current_episode is not None
        return self._current_episode

    @current_episode.setter
    def current_episode(self, episode: Episode) -> None:
        self._current_episode = episode

    @property
    def episode_iterator(self) -> Iterator:
        return self._episode_iterator

    @episode_iterator.setter
    def episode_iterator(self, new_iter: Iterator) -> None:
        self._episode_iterator = new_iter

    @property
    def episodes(self) -> List[Episode]:
        return self._episodes

    @episodes.setter
    def episodes(self, episodes: List[Episode]) -> None:
        assert (
            len(episodes) > 0
        ), "Environment doesn't accept empty episodes list."
        self._episodes = episodes

    @property
    def sim(self) -> Simulator:
        return self._sim

    @property
    def episode_start_time(self) -> Optional[float]:
        return self._episode_start_time

    @property
    def episode_over(self) -> bool:
        return self._episode_over

    @property
    def task(self) -> EmbodiedTask:
        return self._task

    @property
    def _elapsed_seconds(self) -> float:
        assert (
            self._episode_start_time
        ), "Elapsed seconds requested before episode was started."
        return time.time() - self._episode_start_time

    def get_metrics(self) -> Metrics:
        return self._task.measurements.get_metrics()

    def _past_limit(self) -> bool:
        if (
            self._max_episode_steps != 0
            and self._max_episode_steps <= self._elapsed_steps
        ):
            return True
        elif (
            self._max_episode_seconds != 0
            and self._max_episode_seconds <= self._elapsed_seconds
        ):
            return True
        return False

    def _reset_stats(self) -> None:
        self._episode_start_time = time.time()
        self._elapsed_steps = 0
        self._episode_over = False

    def reset(self) -> Observations:
        r"""Resets the environments and returns the initial observations.

        :return: initial observations from the environment.
        """
        self._reset_stats()

        assert len(self.episodes) > 0, "Episodes list is empty"
        # Delete the shortest path cache of the current episode
        # Caching it for the next time we see this episode isn't really worth
        # it
        if self._current_episode is not None:
            self._current_episode._shortest_path_cache = None

        self._current_episode = next(self._episode_iterator)
        self.reconfigure(self._config)

        observations = self.task.reset(episode=self.current_episode)
        self._task.measurements.reset_measures(
            episode=self.current_episode, task=self.task
        )

        return observations

    def _update_step_stats(self) -> None:
        self._elapsed_steps += 1
        self._episode_over = not self._task.is_episode_active
        if self._past_limit():
            self._episode_over = True

        if self.episode_iterator is not None and isinstance(
            self.episode_iterator, EpisodeIterator
        ):
            self.episode_iterator.step_taken()

    def step(
        self, action: Union[int, str, Dict[str, Any]], **kwargs
    ) -> Observations:
        r"""Perform an action in the environment and return observations.

        :param action: action (belonging to :ref:`action_space`) to be
            performed inside the environment. Action is a name or index of
            allowed task's action and action arguments (belonging to action's
            :ref:`action_space`) to support parametrized and continuous
            actions.
        :return: observations after taking action in environment.
        """

        assert (
            self._episode_start_time is not None
        ), "Cannot call step before calling reset"
        assert (
            self._episode_over is False
        ), "Episode over, call reset before calling step"

        # Support simpler interface as well
        if isinstance(action, (str, int, np.integer)):
            action = {"action": action}

        observations = self.task.step(
            action=action, episode=self.current_episode
        )

        self._task.measurements.update_measures(
            episode=self.current_episode, action=action, task=self.task
        )

        self._update_step_stats()

        return observations

    @staticmethod
    @numba.njit
    def _seed_numba(seed: int):
        random.seed(seed)
        np.random.seed(seed)

    def seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self._seed_numba(seed)
        self._sim.seed(seed)
        self._task.seed(seed)

    def reconfigure(self, config: Config) -> None:
        self._config = config

        self._config.defrost()
        self._config.SIMULATOR = self._task.overwrite_sim_config(
            self._config.SIMULATOR, self.current_episode
        )
        self._config.freeze()

        self._sim.reconfigure(self._config.SIMULATOR)

    def render(self, mode="rgb") -> np.ndarray:
        return self._sim.render(mode)

    def close(self) -> None:
        self._sim.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class RLEnv(gym.Env):
    r"""Reinforcement Learning (RL) environment class which subclasses ``gym.Env``.

    This is a wrapper over :ref:`Env` for RL users. To create custom RL
    environments users should subclass `RLEnv` and define the following
    methods: :ref:`get_reward_range()`, :ref:`get_reward()`,
    :ref:`get_done()`, :ref:`get_info()`.

    As this is a subclass of ``gym.Env``, it implements `reset()` and
    `step()`.
    """

    _env: Env

    def __init__(
        self, config: Config, dataset: Optional[Dataset] = None
    ) -> None:
        """Constructor

        :param config: config to construct :ref:`Env`
        :param dataset: dataset to construct :ref:`Env`.
        """

        self._env = Env(config, dataset)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.number_of_episodes = self._env.number_of_episodes
        self.reward_range = self.get_reward_range()

    @property
    def habitat_env(self) -> Env:
        return self._env

    @property
    def episodes(self) -> List[Episode]:
        return self._env.episodes

    @episodes.setter
    def episodes(self, episodes: List[Episode]) -> None:
        self._env.episodes = episodes

    @property
    def current_episode(self) -> Episode:
        return self._env.current_episode

    @profiling_wrapper.RangeContext("RLEnv.reset")
    def reset(self) -> Observations:
        return self._env.reset()

    def get_reward_range(self):
        r"""Get min, max range of reward.

        :return: :py:`[min, max]` range of reward.
        """
        raise NotImplementedError

    def get_reward(self, observations: Observations) -> Any:
        r"""Returns reward after action has been performed.

        :param observations: observations from simulator and task.
        :return: reward after performing the last action.

        This method is called inside the :ref:`step()` method.
        """
        raise NotImplementedError

    def get_done(self, observations: Observations) -> bool:
        r"""Returns boolean indicating whether episode is done after performing
        the last action.

        :param observations: observations from simulator and task.
        :return: done boolean after performing the last action.

        This method is called inside the step method.
        """
        raise NotImplementedError

    def get_info(self, observations) -> Dict[Any, Any]:
        r"""..

        :param observations: observations from simulator and task.
        :return: info after performing the last action.
        """
        raise NotImplementedError

    @profiling_wrapper.RangeContext("RLEnv.step")
    def step(self, *args, **kwargs) -> Tuple[Observations, Any, bool, dict]:
        r"""Perform an action in the environment.

        :return: :py:`(observations, reward, done, info)`
        """

        observations = self._env.step(*args, **kwargs)
        reward = self.get_reward(observations)
        done = self.get_done(observations)
        info = self.get_info(observations)

        return observations, reward, done, info

    def seed(self, seed: Optional[int] = None) -> None:
        self._env.seed(seed)

    def render(self, mode: str = "rgb") -> np.ndarray:
        return self._env.render(mode)

    def close(self) -> None:
        self._env.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# multi-task environment task-change behaviours
class CHANGE_TASK_BEHAVIOUR(enum.IntEnum):
    FIXED = 0
    RANDOM = 1


class CHANGE_TASK_CYCLE_BEHAVIOUR(enum.IntEnum):
    ORDER = 0
    RANDOM = 1


class MultiTaskEnv(Env):
    r"""Environment base class extension designed to handle a stream-line of tasks.

    The major :ref:`task` component is hereby intended as current task with which
    the agent is currently interacting.
    All tasks handled by the MultiTaskEnv are available through the :ref:`tasks` object
    property.

    The environment abstracts away the serving of multiple tasks under
    the scheduling preferences expressed in the configuration.
    """
    _tasks: List[EmbodiedTask]
    DEFAULT_EPS_CHANGE = 10

    def __init__(
        self, config: Config, dataset: Optional[Dataset] = None
    ) -> None:
        """Constructor. Mimics the :ref:`Env` constructor.

        Args:
            :param config: config for the environment. Should contain id for
            simulator and ``task_name`` for each task, which are passed into ``make_sim`` and
            ``make_task``.
            :param dataset: reference to dataset used for first task instance level
            information. Can be defined as :py:`None` in which case
            dataset will be built using ``make_dataset`` and ``config``.
        """
        # let superclass instantiate current task by merging first task in TASKS to TASK
        if len(config.TASKS):
            config.defrost()
            config.TASK.merge_from_other_cfg(config.TASKS[0])
            config.freeze()
        super().__init__(config, dataset=dataset)
        # instatiate other tasks
        self._tasks = [self._task]
        self._curr_task_idx = 0
        for task in config.TASKS[1:]:
            self._tasks.append(
                make_task(
                    task.TYPE,
                    config=task,
                    sim=self._sim,
                    # each task gets its dataset
                    dataset=make_dataset(  # TODO: lazy make_dataset support
                        id_dataset=task.DATASET.TYPE, config=task.DATASET
                    ),
                )
            )
        # episode counter
        self._eps_counter = -1
        self._cumulative_steps_counter = 0
        # when and how to change task is defined here
        self._change_task_behavior = config.CHANGE_TASK_BEHAVIOUR.TYPE
        self._task_cycling_behavior = config.CHANGE_TASK_BEHAVIOUR.LOOP
        # custom task label can be specified
        self._task_label = self._task._config.get(
            "TASK_LABEL", self._curr_task_idx
        )
        # add task_idx to observation space
        self.observation_space = spaces.Dict(
            {
                **self._sim.sensor_suite.observation_spaces.spaces,
                **self._task.sensor_suite.observation_spaces.spaces,
                "task_idx": spaces.Discrete(len(self._tasks)),
            }
        )

    @property
    def tasks(self) -> List[EmbodiedTask]:
        return self._tasks

    @property
    def current_task_label(self) -> str:
        return str(self._task_label)

    def _check_change_task(self, is_reset=False):
        """Check whether the change task condition is satified.
        Args:
            is_reset (bool, optional): Whether we're checking task change condition during a ``reset()``. In that case, we check condition on number of episodes. Defaults to False.
        """
        cum_steps_change = self._config.CHANGE_TASK_BEHAVIOUR.get(
            "AFTER_N_CUM_STEPS", None
        )
        change_ = False
        # check condition on number of episodes only after reset is called
        if is_reset:
            eps_change = self._config.CHANGE_TASK_BEHAVIOUR.get(
                "AFTER_N_EPISODES", None
            )
            if self._change_task_behavior == CHANGE_TASK_BEHAVIOUR.FIXED.name:
                if eps_change and self._eps_counter >= eps_change:
                    change_ = True
                # by default change task every X episodes
                elif (
                    eps_change is None
                    and cum_steps_change is None
                    and self._eps_counter >= eps_change
                ):
                    change_ = True
            elif (
                self._change_task_behavior == CHANGE_TASK_BEHAVIOUR.RANDOM.name
            ):
                # change task with some prob if we exceed min number of episodes
                prob = self._config.CHANGE_TASK_BEHAVIOUR.get(
                    "CHANGE_TASK_PROB", 0.5
                )
                if (
                    eps_change
                    and self._eps_counter >= eps_change
                    and np.random.choice([1, 0], 1, p=[prob, 1 - prob]).item()
                ):
                    change_ = True
                elif (
                    eps_change is None
                    and cum_steps_change is None
                    and self._eps_counter >= eps_change
                    and np.random.choice([1, 0], 1, p=[prob, 1 - prob]).item()
                ):
                    change_ = True

            if change_:
                self._eps_counter = 0
        elif cum_steps_change:
            # episode is still on-going
            if self._change_task_behavior == CHANGE_TASK_BEHAVIOUR.FIXED.name:
                if self._cumulative_steps_counter >= cum_steps_change:
                    change_ = True
            elif (
                self._change_task_behavior == CHANGE_TASK_BEHAVIOUR.RANDOM.name
            ):
                # change task with some prob if we exceed min number of steps
                prob = self._config.CHANGE_TASK_BEHAVIOUR.get(
                    "CHANGE_TASK_PROB", 0.5
                )
                if (
                    self._cumulative_steps_counter >= cum_steps_change
                    and np.random.choice([1, 0], 1, p=[prob, 1 - prob]).item()
                ):
                    change_ = True

            if change_:
                self._cumulative_steps_counter = 0
        return change_

    def _change_task(self, is_reset=False):
        """Change current task according to specified loop behaviour.

        Args:
            is_reset (bool, optional): [description]. Defaults to False.
        """
        prev_task_idx = self._curr_task_idx
        if (
            self._task_cycling_behavior
            == CHANGE_TASK_CYCLE_BEHAVIOUR.ORDER.name
        ):
            self._curr_task_idx = (self._curr_task_idx + 1) % len(self._tasks)
        elif (
            self._task_cycling_behavior
            == CHANGE_TASK_CYCLE_BEHAVIOUR.RANDOM.name
        ):
            # sample from other tasks with equal probability
            self._curr_task_idx = np.random.choice(
                [
                    i
                    for i in range(len(self._tasks))
                    if i != self._curr_task_idx
                ],
                1,
            ).item()

        self._task = self._tasks[self._curr_task_idx]
        # update episode iterator with task dataset
        iter_option_dict = {
            k.lower(): v
            for k, v in self._config.ENVIRONMENT.ITERATOR_OPTIONS.items()
        }
        iter_option_dict["seed"] = self._config.SEED
        self._episode_iterator: EpisodeIterator = (
            self._task._dataset.get_episode_iterator(**iter_option_dict)
        )

        self.action_space = self._task.action_space
        # task observation space may change too
        self.observation_space = spaces.Dict(
            {
                **self._sim.sensor_suite.observation_spaces.spaces,
                **self._task.sensor_suite.observation_spaces.spaces,
                "task_idx": spaces.Discrete(len(self._tasks)),
            }
        )
        self._task_label = self._task._config.get(
            "TASK_LABEL", self._curr_task_idx
        )
        logger.info(
            "Current task changed from {} (id: {}) to {} (id: {}).".format(
                self._tasks[prev_task_idx].__class__.__name__,
                prev_task_idx,
                self._task.__class__.__name__,
                self._curr_task_idx,
            )
        )
        # handle mid-episode change of task
        if not is_reset:
            # keep current position and sim state only if scene does not change
            # if same exact task is passed, you'll get same episode back
            if (
                self.current_episode.scene_id
                == self._episode_iterator.episodes[0].scene_id
            ):
                self._episode_iterator.episodes[
                    0
                ].start_position = self.current_episode.start_position
                self._episode_iterator.episodes[
                    0
                ].start_rotation = self.current_episode.start_rotation
                # self._episode_iterator._iterator = iter(self._episode_iterator.episodes)
                # self.reconfigure(self._config)

                # observations = self.task.reset(episode=self.current_episode)
                # update current episode
                self._current_episode = next(self._episode_iterator)
                # reset prev position
                self._task.measurements.reset_measures(
                    episode=self.current_episode, task=self.task
                )
            else:
                # scene is different, can't keep old position, end episode
                self._episode_over = True

    def step(
        self, action: Union[int, str, Dict[str, Any]], **kwargs
    ) -> Observations:
        obs = super().step(action, **kwargs)
        # check task change behaviour here
        self._cumulative_steps_counter += 1
        if self._check_change_task():
            # task needs to be changed
            self._change_task()

        obs.update({"task_idx": self._curr_task_idx})
        return obs

    def reset(self) -> Observations:
        self._eps_counter += 1
        if self._check_change_task(is_reset=True):
            # task needs to be changed
            self._change_task(is_reset=True)
        # now reset can be called on new task
        obs = super().reset()
        obs["task_label"] = self._curr_task_idx
        return obs


class CRLEnv(RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset]) -> None:
        self._env = MultiTaskEnv(config, dataset)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.number_of_episodes = self._env.number_of_episodes
        self.reward_range = self.get_reward_range()
