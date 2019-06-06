#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""Implements dataset functionality to be used ``habitat.EmbodiedTask``.
``habitat.core.dataset`` abstracts over a collection of 
``habitat.core.Episode``. Each episode consists of a single instantiation
of a ``habitat.Agent`` inside ``habitat.Env``.
"""
import copy
import json
from typing import Callable, Dict, Generic, List, Optional, Type, TypeVar

import attr
import numpy as np

from habitat.core.utils import not_none_validator


@attr.s(auto_attribs=True, kw_only=True)
class Episode:
    r"""Base class for episode specification that includes initial position and
    rotation of agent, scene id, episode. This information is provided by
    a ``Dataset`` instance.

    Args:
        episode_id: id of episode in the dataset, usually episode number.
        scene_id: id of scene in dataset.
        start_position: list of length 3 for cartesian coordinates
            (x, y, z).
        start_rotation: list of length 4 for (x, y, z, w) elements
            of unit quaternion (versor) representing 3D agent orientation
            (https://en.wikipedia.org/wiki/Versor). The rotation specifying
            the agent's orientation is relative to the world coordinate
            axes.
    """

    episode_id: str = attr.ib(default=None, validator=not_none_validator)
    scene_id: str = attr.ib(default=None, validator=not_none_validator)
    start_position: List[float] = attr.ib(
        default=None, validator=not_none_validator
    )
    start_rotation: List[float] = attr.ib(
        default=None, validator=not_none_validator
    )
    info: Optional[Dict[str, str]] = None


T = TypeVar("T", Episode, Type[Episode])


class Dataset(Generic[T]):
    r"""Base class for dataset specification.

    Attributes:
        episodes: list of episodes containing instance information.
    """

    episodes: List[T]

    @property
    def scene_ids(self) -> List[str]:
        r"""
        Returns:
            unique scene ids present in the dataset.
        """
        return sorted(list({episode.scene_id for episode in self.episodes}))

    def get_scene_episodes(self, scene_id: str) -> List[T]:
        r"""
        Args:
            scene_id: id of scene in scene dataset.

        Returns:
            list of episodes for the ``scene_id``.
        """
        return list(
            filter(lambda x: x.scene_id == scene_id, iter(self.episodes))
        )

    def get_episodes(self, indexes: List[int]) -> List[T]:
        r"""
        Args:
            indexes: episode indices in dataset.

        Returns:
            list of episodes corresponding to indexes.
        """
        return [self.episodes[episode_id] for episode_id in indexes]

    def to_json(self) -> str:
        class DatasetJSONEncoder(json.JSONEncoder):
            def default(self, object):
                return object.__dict__

        result = DatasetJSONEncoder().encode(self)
        return result

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        r"""
        Creates dataset from ``json_str``. Directory containing relevant 
        graphical assets of scenes is passed through ``scenes_dir``.

        Args:
            json_str: JSON string containing episodes information.
            scenes_dir: directory containing graphical assets relevant
                for episodes present in ``json_str``.
        """
        raise NotImplementedError

    def filter_episodes(
        self, filter_fn: Callable[[Episode], bool]
    ) -> "Dataset":
        r"""
        Returns a new dataset with only the filtered episodes from the 
        original dataset.

        Args:
            filter_fn: function used to filter the episodes.

        Returns:
            the new dataset.
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
        episodes_per_split: Optional[int] = None,
        remove_unused_episodes: bool = False,
        collate_scene_ids: bool = True,
        sort_by_episode_id: bool = False,
        allow_uneven_splits: bool = False,
    ) -> List["Dataset"]:
        r"""Returns a list of new datasets, each with a subset of the original
        episodes. All splits will have the same number of episodes, but no
        episodes will be duplicated.

        Args:
            num_splits: the number of splits to create.
            episodes_per_split: if provided, each split will have up to
                this many episodes. If it is not provided, each dataset will
                have ``len(original_dataset.episodes) // num_splits`` 
                episodes. If max_episodes_per_split is provided and is 
                larger than this value, it will be capped to this value.
            remove_unused_episodes: once the splits are created, the extra
                episodes will be destroyed from the original dataset. This
                saves memory for large datasets.
            collate_scene_ids: if true, episodes with the same scene id are
                next to each other. This saves on overhead of switching 
                between scenes, but means multiple sequential episodes will 
                be related to each other because they will be in the 
                same scene.
            sort_by_episode_id: if true, sequences are sorted by their episode
                ID in the returned splits.
            allow_uneven_splits: if true, the last split can be shorter than
                the others. This is especially useful for splitting over
                validation/test datasets in order to make sure that all
                episodes are copied but none are duplicated.

        Returns:
            a list of new datasets, each with their own subset of episodes.
        """
        assert (
            len(self.episodes) >= num_splits
        ), "Not enough episodes to create this many splits."
        if episodes_per_split is not None:
            assert not allow_uneven_splits, (
                "You probably don't want to specify allow_uneven_splits"
                " and episodes_per_split."
            )
            assert num_splits * episodes_per_split <= len(self.episodes)

        new_datasets = []

        if allow_uneven_splits:
            stride = int(np.ceil(len(self.episodes) * 1.0 / num_splits))
            split_lengths = [stride] * (num_splits - 1)
            split_lengths.append(
                (len(self.episodes) - stride * (num_splits - 1))
            )
        else:
            if episodes_per_split is not None:
                stride = episodes_per_split
            else:
                stride = len(self.episodes) // num_splits
            split_lengths = [stride] * num_splits

        num_episodes = sum(split_lengths)

        rand_items = np.random.choice(
            len(self.episodes), num_episodes, replace=False
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
            for ii in range(split_lengths[nn]):
                new_dataset.episodes.append(self.episodes[rand_items[ep_ind]])
                ep_ind += 1
            if sort_by_episode_id:
                new_dataset.episodes.sort(key=lambda ep: ep.episode_id)
            new_episodes.extend(new_dataset.episodes)
        if remove_unused_episodes:
            self.episodes = new_episodes
        return new_datasets

    def sample_episodes(self, num_episodes: int) -> None:
        """
        Sample from existing episodes a list of episodes of size num_episodes,
        and replace self.episodes with the list of sampled episodes.
        Args:
            num_episodes: number of episodes to sample, input -1 to use
            whole episodes
        """
        if num_episodes == -1:
            return
        if num_episodes < -1:
            raise ValueError(
                f"Invalid number for episodes to sample: {num_episodes}"
            )
        self.episodes = np.random.choice(
            self.episodes, num_episodes, replace=False
        )
