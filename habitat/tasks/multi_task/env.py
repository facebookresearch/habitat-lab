import enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
from gym import spaces

from habitat.config import Config
from habitat.core.dataset import Dataset, EpisodeIterator
from habitat.core.embodied_task import EmbodiedTask
from habitat.core.env import Env, RLEnv
from habitat.core.logging import logger
from habitat.core.simulator import Observations
from habitat.datasets import make_dataset
from habitat.tasks import make_task


# multi-task environment task-change behaviours
class TI_TASK_CHANGE_TIMESTEP(enum.IntEnum):
    FIXED = 0
    NON_FIXED = 1


class TI_TASK_SAMPLING(enum.IntEnum):
    SEQUENTIAL = 0
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
    _episode_iterators: Optional[List[EpisodeIterator]]

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
        if len(config.MULTI_TASK.TASKS):
            logger.info(
                "Overwriting config.TASK ({}) with first entry in config.MULTI_TASK.TASKS ({}).".format(
                    config.TASK.TYPE, config.MULTI_TASK.TASKS[0].TYPE
                )
            )
            config.defrost()
            config.TASK.merge_from_other_cfg(config.MULTI_TASK.TASKS[0])
            config.freeze()
            # TASKS[0] dataset has higher priority over default one (if specified), instatiate it before
            if dataset is None and config.MULTI_TASK.TASKS[0].DATASET.TYPE:
                dataset = make_dataset(
                    id_dataset=config.MULTI_TASK.TASKS[0].DATASET.TYPE,
                    config=config.MULTI_TASK.TASKS[0].DATASET,
                )
        # initialize first task leveraging Env
        super().__init__(config, dataset=dataset)
        # instatiate other tasks
        self._tasks = [self._task]
        self._curr_task_idx = 0
        # keep each tasks episode iterator to avoid re-creation
        self._episode_iterators = [self._episode_iterator]
        for task in config.MULTI_TASK.TASKS[1:]:
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
            # get task episode iterator
            iter_option_dict = {
                k.lower(): v for k, v in task.EPISODE_ITERATOR_OPTIONS.items()
            }
            iter_option_dict["seed"] = self._config.SEED
            task_ep_iterator: EpisodeIterator = self._tasks[
                -1
            ]._dataset.get_episode_iterator(**iter_option_dict)
            self._episode_iterators.append(task_ep_iterator)
        # episode counter
        self._eps_counter = -1
        self._cumulative_steps_counter = 0
        # when and how to change task is defined here
        self._task_sampling_behavior = (
            config.MULTI_TASK.TASK_ITERATOR.TASK_SAMPLING
        )
        self._change_task_behavior = (
            config.MULTI_TASK.TASK_ITERATOR.TASK_CHANGE_TIMESTEP
        )
        # custom task label can be specified
        self._curr_task_label = self._task._config.get(
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
        """
        Returns underying list of tasks managed by the environment.
        """
        return self._tasks

    @property
    def current_task_label(self) -> str:
        """
        Returns label of the task currently active.
        The label can be set through configuration of a task through the keyword `TASK_LABEL`.
        Alternatively, each task is assigned an index obtained through enumeration of the tasks.
        """
        return str(self._curr_task_label)

    @property
    def episode_iterators(self) -> List[EpisodeIterator]:
        """
        Returns a list of the episode iterators used for episode sampling, one for each task.
        """
        return self._episode_iterators

    @property
    def episode_iterator(self) -> EpisodeIterator:
        return self._episode_iterator

    @episode_iterator.setter
    def episode_iterator(self, new_iter: EpisodeIterator) -> None:
        self._episode_iterator = new_iter
        self._episode_iterators[self._curr_task_idx] = new_iter

    def _check_change_task(self, is_reset=False):
        """Check whether the change task condition is satified.
        Args:
            is_reset (bool, optional): Whether we're checking task change condition during a ``reset()``. In that case, we check condition on number of episodes. Defaults to False.
        """

        def _check_max_task_repeat(
            global_counter: int, counter_threshold: int
        ):
            # checks whether we should change task due to 'episode condition' or `step_condition`, depending
            # on the arguments passed to function
            if (
                self._change_task_behavior
                == TI_TASK_CHANGE_TIMESTEP.FIXED.name
            ):
                # change task at each fixed timestep
                if global_counter >= counter_threshold:
                    return True
            elif (
                self._change_task_behavior
                == TI_TASK_CHANGE_TIMESTEP.NON_FIXED.name
            ):
                # change task with some prob whenever we exceed max number of consecutive
                # episodes
                prob = self._config.MULTI_TASK.TASK_ITERATOR.CHANGE_TASK_PROB
                if (
                    global_counter >= counter_threshold
                    and np.random.choice([1, 0], 1, p=[prob, 1 - prob]).item()
                ):
                    return True
            return False

        cum_steps_change = self._config.MULTI_TASK.TASK_ITERATOR.get(
            "MAX_TASK_REPEAT_STEPS", None
        )
        eps_change = self._config.MULTI_TASK.TASK_ITERATOR.get(
            "MAX_TASK_REPEAT_EPISODES", None
        )
        change_task = False
        # check condition on number of episodes only after reset is called
        if is_reset and eps_change is not None:
            change_task = _check_max_task_repeat(self._eps_counter, eps_change)
            # reset episode counter whenever we switch task
            if change_task:
                self._eps_counter = 0
        elif is_reset and eps_change is None and cum_steps_change is None:
            # when neither max_episodes nor max_steps are defined, we change after X episodes as default
            change_task = _check_max_task_repeat(
                self._eps_counter,
                self._config.MULTI_TASK.TASK_ITERATOR.DEFAULT_MAX_TASK_REPEAT_EPISODES,
            )
            if change_task:
                self._eps_counter = 0
        elif cum_steps_change:
            # episode is still on-going
            change_task = _check_max_task_repeat(
                self._cumulative_steps_counter, cum_steps_change
            )
            if change_task:
                self._cumulative_steps_counter = 0
        return change_task

    def _change_task(self, is_reset=False):
        """Change current task according to specified loop behaviour.

        Args:
            is_reset (bool, optional):  Whether we're changing task during a ``reset()``.
            When false, we need to handle mid-episode change. Defaults to False.
        """
        prev_task_idx = self._curr_task_idx
        if self._task_sampling_behavior == TI_TASK_SAMPLING.SEQUENTIAL.name:
            self._curr_task_idx = (self._curr_task_idx + 1) % len(self._tasks)
        elif self._task_sampling_behavior == TI_TASK_SAMPLING.RANDOM.name:
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
        self._episode_iterator = self._episode_iterators[self._curr_task_idx]

        self.action_space = self._task.action_space
        # task observation space may change too
        self.observation_space = spaces.Dict(
            {
                **self._sim.sensor_suite.observation_spaces.spaces,
                **self._task.sensor_suite.observation_spaces.spaces,
                "task_idx": spaces.Discrete(len(self._tasks)),
            }
        )
        self._curr_task_label = self._task._config.get(
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
