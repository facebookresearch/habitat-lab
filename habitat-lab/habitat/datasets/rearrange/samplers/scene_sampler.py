# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from abc import ABC, abstractmethod
from typing import List


class SceneSampler(ABC):
    @abstractmethod
    def num_scenes(self):
        pass

    def reset(self) -> None:
        pass

    @abstractmethod
    def sample(self):
        pass

    def set_cur_episode(self, cur_episode: int) -> None:
        pass


class SingleSceneSampler(SceneSampler):
    """
    Returns a single provided scene using the sampler API
    """

    def __init__(self, scene: str) -> None:
        self.scene = scene

    def sample(self) -> str:
        return self.scene

    def num_scenes(self) -> int:
        return 1


class MultiSceneSampler(SceneSampler):
    """
    Uniform sampling from a set of scenes.
    """

    def __init__(self, scenes: List[str]) -> None:
        self.scenes = scenes
        assert len(scenes) > 0, "No scenes provided to MultiSceneSampler."

    def sample(self) -> str:
        return self.scenes[random.randrange(0, len(self.scenes))]

    def num_scenes(self) -> int:
        return len(self.scenes)


class BalancedSceneSampler(SceneSampler):
    """
    Evenly splits generated episodes amongst all scenes in the set.
    Generates all episodes for each scene contiguously for efficiency.
    """

    def __init__(self, scenes: List[str], num_episodes: int) -> None:
        assert len(scenes) > 0, "No scenes provided to BalancedSceneSampler."
        self.scenes = scenes
        self.num_episodes = num_episodes
        assert self.num_episodes % len(
            self.scenes
        ) == 0 and self.num_episodes >= len(
            self.scenes
        ), f"Requested number of episodes '{self.num_episodes}' not divisible by number of scenes {len(self.scenes)}, results would be unbalanced."
        self.num_ep_per_scene = int(self.num_episodes / len(self.scenes))
        self.cur_episode = 0

    def sample(self) -> str:
        return self.scenes[int(self.cur_episode / self.num_ep_per_scene)]

    def num_scenes(self) -> int:
        return len(self.scenes)

    def set_cur_episode(self, cur_episode: int) -> None:
        self.cur_episode = cur_episode
