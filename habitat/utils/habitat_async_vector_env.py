from typing import List, Sequence, Tuple, Union

import numpy as np
from gym import spaces
from gym.core import ObsType
from gym.vector import AsyncVectorEnv, VectorEnv, VectorEnvWrapper

from habitat.core.dataset import Episode
from habitat.utils.gym_adapter import HabGymWrapper


class HabitatAsyncVectorEnv(AsyncVectorEnv):
    # TODO : make this AsyncVectorEnv
    def __init__(self, env_fns: Sequence[callable], *args, **kwargs):
        super().__init__(env_fns, *args, **kwargs)
        dummy_env = env_fns[0]()
        self._is_habitat = False
        if isinstance(dummy_env, HabGymWrapper):
            # We need to know if the underlying gym is a habitat environment
            # so we can appropriately query the number of episodes
            self._is_habitat = True
        # _manual_episode_count is a substitute to the habitat environment
        # episode counting in the case the environment is not a habitat env
        self._manual_episode_count = list(range(self.num_envs))
        dummy_env.close()

    @property
    def number_of_episodes(self) -> List[int]:
        """
        Returns the number of episodes the environments hold
        """
        if self._is_habitat:
            return self.call("get_number_of_episodes")
        return [-1] * self.num_envs

    def current_episodes(self) -> List[Episode]:
        """
        Returns the current episode the environments are at
        """
        if self._is_habitat:
            return self.call("get_current_episodes")
        return [
            Episode(
                episode_id=episode,
                scene_id=0,
                start_position=[],
                start_rotation=[],
            )
            for episode in self._manual_episode_count
        ]

    def reset_wait(
        self,
        return_info: bool = False,
        *args,
        **kwargs,
    ) -> Union[ObsType, Tuple[ObsType, List[dict]]]:
        if self._is_habitat:
            return super().reset_wait(return_info, *args, **kwargs)
        max_episode = max(self._manual_episode_count) + 1
        self._manual_episode_count = list(
            range(max_episode, max_episode + self.num_envs)
        )
        return super().reset_wait(return_info, *args, **kwargs)

    def step_wait(
        self, *args, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        if self._is_habitat:
            return super().step_wait(*args, **kwargs)
        _obs, _rew, done, _info = super().step_wait(*args, **kwargs)
        next_episode = max(self._manual_episode_count) + 1
        for i, d in enumerate(done):
            if d:
                self._manual_episode_count[i] = next_episode
                next_episode += 1
        return _obs, _rew, done, _info
