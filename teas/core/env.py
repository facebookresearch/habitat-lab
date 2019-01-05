import time
from typing import Type, List, Tuple, Any, Optional

import gym
from gym.spaces.dict_space import Dict as SpaceDict
import numpy as np

from teas.core.dataset import Dataset, Episode
from teas.core.embodied_task import EmbodiedTask
from teas.core.simulator import Observation
from teas.simulators import make_simulator
from teas.tasks import make_task


class TeasEnv(gym.Env):

    def __init__(self, config: Any, dataset: Optional[Dataset] = None) -> None:
        self._config: Any = config
        self._dataset: Optional[Dataset] = dataset
        self._episodes: List[Type[Episode]] = self._dataset.episodes if \
            self._dataset else []
        self._current_episode_index: Optional[int] = None
        self._simulator = make_simulator(id_simulator=self._config.simulator,
                                         config=self._config)
        self._task: EmbodiedTask = make_task(config.task_name,
                                             config=self._config,
                                             simulator=self._simulator,
                                             dataset=dataset)
        self.observation_space = SpaceDict({
            **self._simulator.sensor_suite.observation_spaces.spaces,
            **self._task.sensor_suite.observation_spaces.spaces
        })
        self.action_space = self._simulator.action_space
        self._max_episode_seconds = getattr(
            self._config, "max_episode_seconds", None)
        self._max_episode_steps = getattr(
            self._config, "max_episode_steps", None)
        self._elapsed_steps = 0
        self._episode_started_at: Optional[float] = None
        self._episode_over = False

    @property
    def current_episode(self) -> Type[Episode]:
        assert self._current_episode_index is not None and \
               self._current_episode_index < len(self._episodes)
        return self._episodes[self._current_episode_index]

    @property
    def episodes(self) -> List[Type[Episode]]:
        return self._episodes

    @episodes.setter
    def episodes(self, episodes: List[Type[Episode]]):
        assert len(
            episodes) > 0, "Environment doesn't accept empty episodes list."
        self._episodes = episodes

    @property
    def _elapsed_seconds(self) -> float:
        assert self._episode_started_at, \
            "Elapsed seconds requested before episode was started."
        return time.time() - self._episode_started_at

    def _past_limit(self) -> bool:
        if self._max_episode_steps is not None and self._max_episode_steps <= \
                self._elapsed_steps:
            return True
        elif self._max_episode_seconds is not None and \
                self._max_episode_seconds <= self._elapsed_seconds:
            return True
        return False

    # TODO(akadian): update the below type hinting after refactor
    def reset(self) -> Tuple[Observation, Observation, bool, None]:
        self._episode_started_at = time.time()
        self._elapsed_steps = 0
        self._episode_over = False

        assert len(self.episodes) > 0, "Episodes list is empty"

        # Switch to next episode in a loop
        if self._current_episode_index is None:
            self._current_episode_index = 0
        else:
            self._current_episode_index = \
                (self._current_episode_index + 1) % len(self._episodes)
        self.reconfigure(self._config)

        # TODO (maksymets) make Task responsible for check if episode is done.
        observations, done = self._simulator.reset()
        observations.update(self._task.sensor_suite.get_observations(
            observations=observations, episode=self.current_episode))
        info = None
        reward = observations["reward"]
        # TODO(akadian, maksymets): decide what to return on reset call
        return observations, reward, done, info

    # TODO(akadian): update the below type hinting after refactor
    def step(self, action: int) -> Tuple[Observation, Observation, bool, None]:
        assert self._episode_started_at is not None, "Cannot call step " \
                                                     "before calling reset"
        assert self._episode_over is False, "Episode done, call reset " \
                                            "before calling step"

        observations, done = self._simulator.step(action)
        observations.update(
            self._task.sensor_suite.get_observations(
                observations=observations,
                episode=self.current_episode))

        self._elapsed_steps += 1

        if self._past_limit():
            done = True
            self._episode_over = True

        info = None
        reward = observations["reward"]
        # TODO(akadian): move away the below 4 tuple to RL environment
        return observations, reward, done, info

    def seed(self, seed: int = None) -> None:
        self._simulator.seed(seed)

    def reconfigure(self, config) -> None:
        # TODO (maksymets) switch to self._config.simulator when it will
        #  be separated
        self._config = self._task.overwrite_sim_config(self._config,
                                                       self.current_episode)
        self._simulator.reconfigure(config)

    def geodesic_distance(self, position_a, position_b) -> float:
        return self._simulator.geodesic_distance(position_a, position_b)

    def semantic_annotations(self):
        return self._simulator.semantic_annotations()

    def render(self, mode='human', close=False) -> np.ndarray:
        return self._simulator.render(mode, close)

    def close(self) -> None:
        self._simulator.close()
