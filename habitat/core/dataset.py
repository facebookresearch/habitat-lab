#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import Dict, List, Type, TypeVar, Generic, Optional


class Episode:
    """Base class for episode specification that includes initial position and
    rotation of agent, scene id, episode. This information is provided by
    a Dataset instance.

    Args:
        episode_id: id of episode in the dataset, usually episode number
        scene_id: id of scene in dataset.
        start_position: list of length 3 for cartesian coordinates
            (x, y, z).
        start_rotation: list of length 4 for (x, y, z, w) elements
            of unit quaternion (versor) representing 3D agent orientation
            (https://en.wikipedia.org/wiki/Versor). The rotation specifying
            the agent's orientation is relative to the world coordinate
            axes.
    """

    episode_id: str
    scene_id: str
    start_position: List[float]
    start_rotation: List[float]
    info: Optional[Dict[str, str]] = None

    def __init__(
        self,
        episode_id: str,
        scene_id: str,
        start_position: List[float],
        start_rotation: List[float],
        info: Optional[Dict[str, str]] = None,
    ) -> None:
        self.episode_id = episode_id
        self.scene_id = scene_id
        self.start_position = start_position
        self.start_rotation = start_rotation
        self.info = info

    def __str__(self):
        return str(self.__dict__)


T = TypeVar("T", Episode, Type[Episode])


class Dataset(Generic[T]):
    """Base class for dataset specification.

    Attributes:
        episodes: list of episodes containing instance information
    """

    episodes: List[T]

    @property
    def scene_ids(self) -> List[str]:
        """
        Returns:
            unique scene ids present in the dataset
        """
        return list({episode.scene_id for episode in self.episodes})

    def get_scene_episodes(self, scene_id: str) -> List[T]:
        """
        Args:
            scene_id: id of scene in scene dataset

        Returns:
            list of episodes for the scene_id
        """
        return list(
            filter(lambda x: x.scene_id == scene_id, iter(self.episodes))
        )

    def get_episodes(self, indexes: List[int]) -> List[T]:
        """
        Args:
            indexes: episode indices in dataset

        Returns:
            list of episodes corresponding to indexes
        """
        return [self.episodes[episode_id] for episode_id in indexes]

    def to_json(self) -> str:
        class DatasetJSONEncoder(json.JSONEncoder):
            def default(self, object):
                return object.__dict__

        result = DatasetJSONEncoder().encode(self)
        return result

    def from_json(self, json_str: str) -> None:
        raise NotImplementedError
