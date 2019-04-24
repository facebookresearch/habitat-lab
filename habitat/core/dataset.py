#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import random
from typing import Dict, List, Type, TypeVar, Generic, Optional, Callable

import numpy as np


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
        return sorted(list({episode.scene_id for episode in self.episodes}))

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

    def filter_episodes(
        self, filter_fn: Callable[[Episode], bool]
    ) -> "Dataset":
        """
        Returns a new dataset with only the filtered episodes from the original
        dataset.
        Args:
            filter_fn: Function used to filter the episodes.
        Returns:
            The new dataset.
        """
        new_episodes = []
        for episode in self.episodes:
            if filter_fn(episode):
                new_episodes.append(episode)
        new_dataset = copy.copy(self)
        new_dataset.episodes = new_episodes
        return new_dataset

    def get_splits(
        self,
        num_splits: int,
        max_episodes_per_split: Optional[int] = None,
        remove_unused_episodes: bool = False,
        collate_scene_ids: bool = True,
        sort_by_episode_id: bool = False,
    ) -> List["Dataset"]:
        """
        Returns a list of new datasets, each with a subset of the original
        episodes. All splits will have the same number of episodes, but no
        episodes will be duplicated.
        Args:
            num_splits: The number of splits to create.
            max_episodes_per_split: If provided, each split will have up to
                this many episodes. If it is not provided, each dataset will
                have len(original_dataset.episodes) // num_splits episodes. If
                max_episodes_per_split is provided and is larger than this
                value, it will be capped to this value.
            remove_unused_episodes: Once the splits are created, the extra
                episodes will be destroyed from the original dataset. This
                saves memory for large datasets.
            collate_scene_ids: If true, episodes with the same scene id are
                next to each other. This saves on overhead of switching between
                scenes, but means multiple sequential episodes will be related
                to each other because they will be in the same scene.
            sort_by_episode_id: If true, sequences are sorted by their episode
                ID in the returned splits.
        Returns:
            A list of new datasets, each with their own subset of episodes.
        """

        assert (
            len(self.episodes) >= num_splits
        ), "Not enough episodes to create this many splits."

        new_datasets = []
        if max_episodes_per_split is None:
            max_episodes_per_split = len(self.episodes) // num_splits
        max_episodes_per_split = min(
            max_episodes_per_split, (len(self.episodes) // num_splits)
        )
        rand_items = np.random.choice(
            len(self.episodes),
            num_splits * max_episodes_per_split,
            replace=False,
        )
        if collate_scene_ids:
            scene_ids = {}
            for rand_ind in rand_items:
                scene = self.episodes[rand_ind].scene_id
                if scene not in scene_ids:
                    scene_ids[scene] = []
                scene_ids[scene].append(rand_ind)
            rand_items = []
            list(map(rand_items.extend, scene_ids.values()))
        ep_ind = 0
        new_episodes = []
        for nn in range(num_splits):
            new_dataset = copy.copy(self)  # Creates a shallow copy
            new_dataset.episodes = []
            new_datasets.append(new_dataset)
            for ii in range(max_episodes_per_split):
                new_dataset.episodes.append(self.episodes[rand_items[ep_ind]])
                ep_ind += 1
            if sort_by_episode_id:
                new_dataset.episodes.sort(key=lambda ep: ep.episode_id)
            new_episodes.extend(new_dataset.episodes)
        if remove_unused_episodes:
            self.episodes = new_episodes
        return new_datasets

    def get_uneven_splits(self, num_splits):
        """
        Returns a list of new datasets, each with a subset of the original
        episodes. The last dataset may have fewer episodes than the others.
        This is especially useful for splitting over validation/test datasets
        in order to make sure that all episodes are copied but none are
        duplicated.
        Args:
            num_splits: The number of splits to create.
        Returns:
            A list of new datasets, each with their own subset of episodes.
        """
        assert (
            len(self.episodes) >= num_splits
        ), "Not enough episodes to create this many splits."
        new_datasets = []
        num_episodes = len(self.episodes)
        stride = int(np.ceil(num_episodes * 1.0 / num_splits))
        for ii, split in enumerate(
            range(0, num_episodes, stride)[:num_splits]
        ):
            new_dataset = copy.copy(self)  # Creates a shallow copy
            new_dataset.episodes = new_dataset.episodes[
                split : min(split + stride, num_episodes)
            ].copy()
            new_datasets.append(new_dataset)
        assert (
            sum([len(new_dataset.episodes) for new_dataset in new_datasets])
            == num_episodes
        )
        return new_datasets
