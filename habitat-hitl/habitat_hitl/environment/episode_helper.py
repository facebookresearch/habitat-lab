#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

import habitat
from habitat.core.dataset import Episode, EpisodeIterator


class EpisodeHelper:
    def __init__(self, habitat_env: habitat.Env) -> None:
        self._habitat_env: habitat.Env = habitat_env
        self._episode_iterator: EpisodeIterator = habitat_env.episode_iterator  # type: ignore
        self._num_iter_episodes: Union[int, float] = (
            len(self._episode_iterator.episodes)
            if not self._episode_iterator.cycle
            else float("inf")
        )
        self._num_episodes_done: int = 0

    @property
    def num_iter_episodes(self) -> Union[int, float]:
        return self._num_iter_episodes

    @property
    def num_episodes_done(self) -> int:
        return self._num_episodes_done

    @property
    def current_episode(self) -> Episode:
        return self._habitat_env.current_episode

    def set_next_episode_by_index(self, episode_index: int) -> None:
        """
        Set the next episode to run by episode index.
        The new episode will be loading upon resetting the simulator.
        """
        self._episode_iterator.set_next_episode_by_index(episode_index)

    def set_next_episode_by_id(self, episode_id: str) -> None:
        """
        Set the next episode to run by episode ID.
        The new episode will be loading upon resetting the simulator.
        """
        self._episode_iterator.set_next_episode_by_id(episode_id)

    def next_episode_exists(self) -> bool:
        return self._num_episodes_done < self._num_iter_episodes - 1

    def increment_done_episode_counter(self) -> None:
        self._num_episodes_done += 1
        assert self._num_episodes_done <= self._num_iter_episodes
