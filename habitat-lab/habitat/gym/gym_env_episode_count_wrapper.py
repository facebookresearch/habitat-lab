# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Union

from gym import Env, Wrapper, spaces
from gym.core import ActType, ObsType

from habitat.core.dataset import Episode


class EnvCountEpisodeWrapper(Wrapper):
    OBSERVATION_KEY = "obs"
    observation_space: spaces.Space

    def __init__(self, env: Env):
        """
        A helper wrapper to count the number of episodes available
        """
        super().__init__(env)
        self._has_number_episode = hasattr(env, "number_of_episodes")
        self._current_episode = 0

    @property
    def number_of_episodes(self):
        if self._has_number_episode:
            return self.env.number_of_episodes
        else:
            return -1

    @property
    def current_episode(self) -> Episode:
        if self._has_number_episode:
            return self.env.current_episode
        else:
            return Episode(
                episode_id=str(self._current_episode),
                scene_id="default",
                start_position=[],
                start_rotation=[],
            )

    @property
    def original_action_space(self) -> spaces.space:
        if self._has_number_episode:
            return self.env.original_action_space
        else:
            return self.action_space

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        """Steps through the environment with action."""
        o, r, done, i = self.env.step(action)
        if done:
            self._current_episode += 1
        return o, r, done, i

    def reset(self, **kwargs) -> Union[ObsType, Tuple[ObsType, dict]]:
        """Resets the environment with kwargs."""
        self._current_episode += 1
        return self.env.reset(**kwargs)
