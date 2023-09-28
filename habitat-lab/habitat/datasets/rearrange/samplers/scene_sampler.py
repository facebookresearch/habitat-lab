# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from abc import ABC, abstractmethod
from typing import List


class SceneSampler(ABC):
    """
    Abstract Class
    Samples a scene for the RearrangeGenerator.
    """

    @abstractmethod
    def num_scenes(self):
        """
        Get the number of scenes available from this sampler.
        """

    @abstractmethod
    def sample(self):
        """
        Sample a scene.
        """

    def set_cur_episode(self, cur_episode: int) -> None:
        """
        Set the current episode index. Used by some sampler implementations which pivot on the total number of successful episodes generated thus far.
        """


class SingleSceneSampler(SceneSampler):
    """
    Returns a single provided scene using the sampler API
    """

    def __init__(self, scene: str) -> None:
        self.scene = scene

    def sample(self) -> str:
        return self.scene

    def num_scenes(self) -> int:
        """
        Get the number of scenes available from this sampler.
        Single scene sampler always has 1 scene.
        """
        return 1


class MultiSceneSampler(SceneSampler):
    """
    Uniform sampling from a set of scenes.
    """

    def __init__(self, scenes: List[str]) -> None:
        # ensure uniqueness
        self.scenes = list(set(scenes))
        assert len(scenes) > 0, "No scenes provided to MultiSceneSampler."

    def sample(self) -> str:
        """
        Sample a random scene from the configured set.
        """
        return self.scenes[random.randrange(0, len(self.scenes))]

    def num_scenes(self) -> int:
        """
        Get the number of scenes available from this sampler.
        Total number of unique scenes available in all provided scene sets.
        """
        return len(self.scenes)


class BalancedSceneSampler(SceneSampler):
    """
    Evenly splits generated episodes amongst all scenes in the set.
    Generates all episodes for each scene contiguously for efficiency.
    """

    def __init__(self, scenes: List[str], num_episodes: int) -> None:
        """
        Initialize the BalancedSceneSampler for a pre-determined number of episodes.
        This number must be accurate for correct behavior.
        """
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
        """
        Return the next scene in the sequence based on current episode index.
        """
        return self.scenes[int(self.cur_episode / self.num_ep_per_scene)]

    def num_scenes(self) -> int:
        """
        Get the number of scenes available from this sampler.
        """
        return len(self.scenes)

    def set_cur_episode(self, cur_episode: int) -> None:
        """
        Set the current episode index.
        Determines which scene in the sequence to sample.
        Must be strictly less than the configured num_episodes.
        """
        self.cur_episode = cur_episode
        assert (
            self.cur_episode <= self.num_episodes
        ), f"Current episode index {self.cur_episode} is out of initially configured range {self.num_episodes}. BalancedSceneSampler behavior is not defined in these conditions. Initially configured number of episodes must be accurate."
