#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class EpisodeHelper:
    def __init__(self, habitat_env):
        self._habitat_env = habitat_env

        self._num_iter_episodes: int = len(self._habitat_env.episode_iterator.episodes)  # type: ignore
        self._num_episodes_done: int = 0

    @property
    def num_iter_episodes(self):
        return self._num_iter_episodes

    @property
    def num_episodes_done(self):
        return self._num_episodes_done

    @property
    def current_episode(self):
        return self._habitat_env.current_episode

    def next_episode_exists(self):
        return self._num_episodes_done < self._num_iter_episodes - 1

    def increment_done_episode_counter(self):
        self._num_episodes_done += 1
        assert self._num_episodes_done <= self._num_iter_episodes
