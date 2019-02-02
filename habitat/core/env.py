import time
from typing import Dict, Type, List, Any, Optional, Tuple

import gym
import numpy as np
from gym.spaces.dict_space import Dict as SpaceDict

from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import EmbodiedTask
from habitat.core.simulator import AgentState, ShortestPathPoint
from habitat.core.simulator import Observations, Simulator
from habitat.sims import make_sim
from habitat.tasks import make_task


class Env:
    observation_space: SpaceDict
    action_space: SpaceDict
    _config: Config
    _dataset: Optional[Dataset]
    _episodes: List[Type[Episode]]
    _current_episode_index: Optional[int]
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
        self._config = config.clone()
        self._dataset = dataset
        self._episodes = self._dataset.episodes if self._dataset else []
        self._current_episode_index = None
        self._sim = make_sim(
            id_sim=self._config.SIMULATOR.TYPE, config=self._config.SIMULATOR
        )
        self._task = make_task(
            self._config.TASK.TYPE,
            config=self._config.TASK,
            sim=self._sim,
            dataset=dataset,
        )
        self.observation_space = SpaceDict(
            {
                **self._sim.sensor_suite.observation_spaces.spaces,
                **self._task.sensor_suite.observation_spaces.spaces,
            }
        )
        self.action_space = self._sim.action_space
        self._max_episode_seconds = (
            self._config.ENVIRONMENT.MAX_EPISODE_SECONDS
        )
        self._max_episode_steps = self._config.ENVIRONMENT.MAX_EPISODE_STEPS
        self._elapsed_steps = 0
        self._episode_start_time: Optional[float] = None
        self._episode_over = False

    @property
    def current_episode(self) -> Type[Episode]:
        assert (
            self._current_episode_index is not None
            and self._current_episode_index < len(self._episodes)
        )
        return self._episodes[self._current_episode_index]

    @property
    def episodes(self) -> List[Type[Episode]]:
        return self._episodes

    @episodes.setter
    def episodes(self, episodes: List[Type[Episode]]) -> None:
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
        self._reset_stats()

        assert len(self.episodes) > 0, "Episodes list is empty"

        # Switch to next episode in a loop
        if self._current_episode_index is None:
            self._current_episode_index = 0
        else:
            self._current_episode_index = (
                self._current_episode_index + 1
            ) % len(self._episodes)
        self.reconfigure(self._config)

        observations = self._sim.reset()
        observations.update(
            self.task.sensor_suite.get_observations(
                observations=observations, episode=self.current_episode
            )
        )

        return observations

    def _update_step_stats(self) -> None:
        self._elapsed_steps += 1
        self._episode_over = not self._sim.is_episode_active
        if self._past_limit():
            self._episode_over = True

    def step(self, action: int) -> Observations:
        assert self._episode_start_time is not None, (
            "Cannot call step " "before calling reset"
        )
        assert self._episode_over is False, (
            "Episode over, call reset " "before calling step"
        )

        observations = self._sim.step(action)
        observations.update(
            self._task.sensor_suite.get_observations(
                observations=observations, episode=self.current_episode
            )
        )

        self._update_step_stats()

        return observations

    def seed(self, seed: int) -> None:
        self._sim.seed(seed)

    def reconfigure(self, config: Config) -> None:
        self._config = config.clone()
        self._config.SIMULATOR = self._task.overwrite_sim_config(
            self._config.SIMULATOR, self.current_episode
        )
        self._sim.reconfigure(self._config.SIMULATOR)

    def geodesic_distance(
        self, position_a: List[float], position_b: List[float]
    ) -> float:
        return self._sim.geodesic_distance(position_a, position_b)

    def semantic_annotations(self):
        return self._sim.semantic_annotations()

    def sample_navigable_point(self) -> List[float]:
        return self._sim.sample_navigable_point()

    def action_space_shortest_path(
        self, source: AgentState, targets: List[AgentState]
    ) -> List[ShortestPathPoint]:
        r"""
        :param source: source agent state for shortest path calculation
        :param targets: target agent state(s) for shortest path calculation
        :return: List of agent states and actions along the shortest path from
        source to the nearest target (both included). If one of the target(s)
        is identical to the source, a list containing only one node with the
        identical agent state is returned. Returns an empty list in case none
        of the targets are reachable from the source.
        """
        return self._sim.action_space_shortest_paths(
            source, targets, agent_id=0
        )

    def render(self, mode="human", close=False) -> np.ndarray:
        return self._sim.render(mode, close)

    def close(self) -> None:
        self._sim.close()


class RLEnv(gym.Env):
    _env: Env

    def __init__(
        self, config: Config, dataset: Optional[Dataset] = None
    ) -> None:
        self._env = Env(config, dataset)

    @property
    def habitat_env(self) -> Env:
        return self._env

    @property
    def episodes(self) -> List[Type[Episode]]:
        return self._env.episodes

    @episodes.setter
    def episodes(self, episodes: List[Type[Episode]]) -> None:
        self._env.episodes = episodes

    def reset(self) -> Observations:
        observations = self._env.reset()

        return observations

    def get_reward(self, observations: Observations) -> Any:
        raise NotImplementedError

    def get_done(self, observations: Observations) -> bool:
        return self._env.episode_over

    def get_info(self, observations) -> Dict[Any, Any]:
        raise NotImplementedError

    def step(self, action: int) -> Tuple[Observations, Any, bool, dict]:
        assert self._env.episode_start_time is not None, (
            "Cannot call step " "before calling reset"
        )
        assert self._env.episode_over is False, (
            "Episode over,  call reset " "before calling step"
        )

        observations = self._env.step(action)

        reward = self.get_reward(observations)
        done = self.get_done(observations)
        info = self.get_info(observations)

        return observations, reward, done, info

    def seed(self, seed: int) -> None:
        self._env.seed(seed)

    def render(self, mode: str = "human", close: bool = False) -> np.ndarray:
        return self._env.render(mode, close)

    def close(self) -> None:
        self._env.close()
