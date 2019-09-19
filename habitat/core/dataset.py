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
import random
from itertools import groupby
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    TypeVar,
)

import attr
import numpy as np

from habitat.core.utils import not_none_validator


@attr.s(auto_attribs=True, kw_only=True)
class Episode:
    r"""Base class for episode specification that includes initial position and
    rotation of agent, scene id, episode.

    :property episode_id: id of episode in the dataset, usually episode number.
    :property scene_id: id of scene in dataset.
    :property start_position: list of length 3 for cartesian coordinates
        :py:`(x, y, z)`.
    :property start_rotation: list of length 4 for (x, y, z, w) elements
        of unit quaternion (versor) representing 3D agent orientation
        (https://en.wikipedia.org/wiki/Versor). The rotation specifying the
        agent's orientation is relative to the world coordinate axes.

    This information is provided by a `Dataset` instance.
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


T = TypeVar("T", bound=Episode)


class Dataset(Generic[T]):
    r"""Base class for dataset specification.
    """
    episodes: List[T]

    @property
    def num_episodes(self) -> int:
        r"""number of episodes in the dataset
        """
        return len(self.episodes)

    @property
    def scene_ids(self) -> List[str]:
        r"""unique scene ids present in the dataset.
        """
        return sorted(list({episode.scene_id for episode in self.episodes}))

    def get_scene_episodes(self, scene_id: str) -> List[T]:
        r"""..

        :param scene_id: id of scene in scene dataset.
        :return: list of episodes for the :p:`scene_id`.
        """
        return list(
            filter(lambda x: x.scene_id == scene_id, iter(self.episodes))
        )

    def get_episodes(self, indexes: List[int]) -> List[T]:
        r"""..

        :param indexes: episode indices in dataset.
        :return: list of episodes corresponding to indexes.
        """
        return [self.episodes[episode_id] for episode_id in indexes]

    def get_episode_iterator(self, *args: Any, **kwargs: Any) -> Iterator:
        r"""Gets episode iterator with options. Options are specified in
        `EpisodeIterator` documentation.

        :param args: positional args for iterator constructor
        :param kwargs: keyword args for iterator constructor
        :return: episode iterator with specified behavior

        To further customize iterator behavior for your `Dataset` subclass,
        create a customized iterator class like `EpisodeIterator` and override
        this method.
        """
        return EpisodeIterator(self.episodes, *args, **kwargs)

    def to_json(self) -> str:
        class DatasetJSONEncoder(json.JSONEncoder):
            def default(self, object):
                return object.__dict__

        result = DatasetJSONEncoder().encode(self)
        return result

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        r"""Creates dataset from :p:`json_str`.

        :param json_str: JSON string containing episodes information.
        :param scenes_dir: directory containing graphical assets relevant
            for episodes present in :p:`json_str`.

        Directory containing relevant graphical assets of scenes is passed
        through :p:`scenes_dir`.
        """
        raise NotImplementedError

    def filter_episodes(self, filter_fn: Callable[[T], bool]) -> "Dataset":
        r"""Returns a new dataset with only the filtered episodes from the
        original dataset.

        :param filter_fn: function used to filter the episodes.
        :return: the new dataset.
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
        episodes.

        :param num_splits: the number of splits to create.
        :param episodes_per_split: if provided, each split will have up to this
            many episodes. If it is not provided, each dataset will have
            :py:`len(original_dataset.episodes) // num_splits` episodes. If
            max_episodes_per_split is provided and is larger than this value,
            it will be capped to this value.
        :param remove_unused_episodes: once the splits are created, the extra
            episodes will be destroyed from the original dataset. This saves
            memory for large datasets.
        :param collate_scene_ids: if true, episodes with the same scene id are
            next to each other. This saves on overhead of switching between
            scenes, but means multiple sequential episodes will be related to
            each other because they will be in the same scene.
        :param sort_by_episode_id: if true, sequences are sorted by their
            episode ID in the returned splits.
        :param allow_uneven_splits: if true, the last splits can be shorter
            than the others. This is especially useful for splitting over
            validation/test datasets in order to make sure that all episodes
            are copied but none are duplicated.
        :return: a list of new datasets, each with their own subset of
            episodes.

        All splits will have the same number of episodes, but no episodes will
        be duplicated.
        """
        if self.num_episodes < num_splits:
            raise ValueError(
                "Not enough episodes to create those many splits."
            )

        if episodes_per_split is not None:
            if allow_uneven_splits:
                raise ValueError(
                    "You probably don't want to specify allow_uneven_splits"
                    " and episodes_per_split."
                )

            if num_splits * episodes_per_split > self.num_episodes:
                raise ValueError(
                    "Not enough episodes to create those many splits."
                )

        new_datasets = []

        if episodes_per_split is not None:
            stride = episodes_per_split
        else:
            stride = self.num_episodes // num_splits
        split_lengths = [stride] * num_splits

        if allow_uneven_splits:
            episodes_left = self.num_episodes - stride * num_splits
            split_lengths[:episodes_left] = [stride + 1] * episodes_left
            assert sum(split_lengths) == self.num_episodes

        num_episodes = sum(split_lengths)

        rand_items = np.random.choice(
            self.num_episodes, num_episodes, replace=False
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


class EpisodeIterator(Iterator):
    r"""Episode Iterator class that gives options for how a list of episodes
    should be iterated.

    Some of those options are desirable for the internal simulator to get
    higher performance. More context: simulator suffers overhead when switching
    between scenes, therefore episodes of the same scene should be loaded
    consecutively. However, if too many consecutive episodes from same scene
    are feed into RL model, the model will risk to overfit that scene.
    Therefore it's better to load same scene consecutively and switch once a
    number threshold is reached.

    Currently supports the following features:

    Cycling:
        when all episodes are iterated, cycle back to start instead of throwing
        StopIteration.
    Cycling with shuffle:
        when cycling back, shuffle episodes groups grouped by scene.
    Group by scene:
        episodes of same scene will be grouped and loaded consecutively.
    Set max scene repeat:
        set a number threshold on how many episodes from the same scene can be
        loaded consecutively.
    Sample episodes:
        sample the specified number of episodes.
    """

    def __init__(
        self,
        episodes: List[T],
        cycle: bool = True,
        shuffle: bool = False,
        group_by_scene: bool = True,
        max_scene_repeat: int = -1,
        num_episode_sample: int = -1,
    ):
        r"""..

        :param episodes: list of episodes.
        :param cycle: if :py:`True`, cycle back to first episodes when
            StopIteration.
        :param shuffle: if :py:`True`, shuffle scene groups when cycle. No
            effect if cycle is set to :py:`False`. Will shuffle grouped scenes
            if :p:`group_by_scene` is :py:`True`.
        :param group_by_scene: if :py:`True`, group episodes from same scene.
        :param max_scene_repeat: threshold of how many episodes from the same
            scene can be loaded consecutively. :py:`-1` for no limit
        :param num_episode_sample: number of episodes to be sampled. :py:`-1`
            for no sampling.
        """
        # sample episodes
        if num_episode_sample >= 0:
            episodes = np.random.choice(
                episodes, num_episode_sample, replace=False
            )
        self.episodes = episodes
        self.cycle = cycle
        self.group_by_scene = group_by_scene
        if group_by_scene:
            num_scene_groups = len(
                list(groupby(episodes, key=lambda x: x.scene_id))
            )
            num_unique_scenes = len(set([e.scene_id for e in episodes]))
            if num_scene_groups >= num_unique_scenes:
                self.episodes = sorted(self.episodes, key=lambda x: x.scene_id)
        self.max_scene_repetition = max_scene_repeat
        self.shuffle = shuffle
        self._rep_count = 0
        self._prev_scene_id = None
        self._iterator = iter(self.episodes)

    def __iter__(self):
        return self

    def __next__(self):
        r"""The main logic for handling how episodes will be iterated.

        :return: next episode.
        """

        next_episode = next(self._iterator, None)
        if next_episode is None:
            if not self.cycle:
                raise StopIteration
            self._iterator = iter(self.episodes)
            if self.shuffle:
                self._shuffle_iterator()
            next_episode = next(self._iterator)

        if self._prev_scene_id == next_episode.scene_id:
            self._rep_count += 1
        if (
            self.max_scene_repetition > 0
            and self._rep_count >= self.max_scene_repetition - 1
        ):
            self._shuffle_iterator()
            self._rep_count = 0

        self._prev_scene_id = next_episode.scene_id
        return next_episode

    def _shuffle_iterator(self) -> None:
        r"""Internal method that shuffles the remaining episodes.
            If self.group_by_scene is true, then shuffle groups of scenes.
        """
        if self.group_by_scene:
            grouped_episodes = [
                list(g)
                for k, g in groupby(self._iterator, key=lambda x: x.scene_id)
            ]
            random.shuffle(grouped_episodes)
            self._iterator = iter(sum(grouped_episodes, []))
        else:
            episodes = list(self._iterator)
            random.shuffle(episodes)
            self._iterator = iter(episodes)
