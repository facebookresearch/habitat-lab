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
