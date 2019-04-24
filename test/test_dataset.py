#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.dataset import Dataset, Episode


def _construct_dataset(num_episodes):
    episodes = []
    for ii in range(num_episodes):
        episode = Episode(
            episode_id=str(ii),
            scene_id="scene_id_" + str(ii % 10),
            start_position=[0, 0, 0],
            start_rotation=[0, 0, 0, 1],
        )
        episodes.append(episode)
    dataset = Dataset()
    dataset.episodes = episodes
    return dataset


def test_scene_ids():
    dataset = _construct_dataset(100)
    assert dataset.scene_ids == ["scene_id_" + str(ii) for ii in range(10)]


def test_get_scene_episodes():
    dataset = _construct_dataset(100)
    scene = "scene_id_0"
    scene_episodes = dataset.get_scene_episodes(scene)
    assert len(scene_episodes) == 10
    for ep in scene_episodes:
        assert ep.scene_id == scene


def test_filter_episodes():
    dataset = _construct_dataset(100)

    def filter_fn(episode: Episode) -> bool:
        return int(episode.episode_id) % 2 == 0

    filtered_dataset = dataset.filter_episodes(filter_fn)
    assert len(filtered_dataset.episodes) == 50
    for ep in filtered_dataset.episodes:
        assert filter_fn(ep)


def test_get_splits_even_split_possible():
    dataset = _construct_dataset(100)
    splits = dataset.get_splits(10)
    assert len(splits) == 10
    for split in splits:
        assert len(split.episodes) == 10


def test_get_splits_with_remainder():
    dataset = _construct_dataset(100)
    splits = dataset.get_splits(11)
    assert len(splits) == 11
    for split in splits:
        assert len(split.episodes) == 9


def test_get_splits_max_episodes_specified():
    dataset = _construct_dataset(100)
    splits = dataset.get_splits(10, 3, False)
    assert len(splits) == 10
    for split in splits:
        assert len(split.episodes) == 3
    assert len(dataset.episodes) == 100

    dataset = _construct_dataset(100)
    splits = dataset.get_splits(10, 11, False)
    assert len(splits) == 10
    for split in splits:
        assert len(split.episodes) == 10
    assert len(dataset.episodes) == 100

    dataset = _construct_dataset(100)
    splits = dataset.get_splits(10, 3, True)
    assert len(splits) == 10
    for split in splits:
        assert len(split.episodes) == 3
    assert len(dataset.episodes) == 30


def test_get_splits_collate_scenes():
    dataset = _construct_dataset(10000)
    splits = dataset.get_splits(10, 23, collate_scene_ids=True)
    assert len(splits) == 10
    for split in splits:
        assert len(split.episodes) == 23
        prev_ids = set()
        for ii, ep in enumerate(split.episodes):
            if ep.scene_id not in prev_ids:
                prev_ids.add(ep.scene_id)
            else:
                assert split.episodes[ii - 1].scene_id == ep.scene_id

    dataset = _construct_dataset(10000)
    splits = dataset.get_splits(10, 200, collate_scene_ids=False)
    assert len(splits) == 10
    for split in splits:
        prev_ids = set()
        found_not_collated = False
        for ii, ep in enumerate(split.episodes):
            if ep.scene_id not in prev_ids:
                prev_ids.add(ep.scene_id)
            else:
                if split.episodes[ii - 1].scene_id != ep.scene_id:
                    found_not_collated = True
                    break
        assert found_not_collated

    dataset = _construct_dataset(10000)
    splits = dataset.get_splits(10, collate_scene_ids=True)
    assert len(splits) == 10
    for split in splits:
        assert len(split.episodes) == 1000
        prev_ids = set()
        for ii, ep in enumerate(split.episodes):
            if ep.scene_id not in prev_ids:
                prev_ids.add(ep.scene_id)
            else:
                assert split.episodes[ii - 1].scene_id == ep.scene_id

    dataset = _construct_dataset(10000)
    splits = dataset.get_splits(10, collate_scene_ids=False)
    assert len(splits) == 10
    for split in splits:
        prev_ids = set()
        found_not_collated = False
        for ii, ep in enumerate(split.episodes):
            if ep.scene_id not in prev_ids:
                prev_ids.add(ep.scene_id)
            else:
                if split.episodes[ii - 1].scene_id != ep.scene_id:
                    found_not_collated = True
                    break
        assert found_not_collated


def test_get_splits_sort_by_episode_id():
    dataset = _construct_dataset(10000)
    splits = dataset.get_splits(10, 23, sort_by_episode_id=True)
    assert len(splits) == 10
    for split in splits:
        assert len(split.episodes) == 23
        for ii, ep in enumerate(split.episodes):
            if ii > 0:
                assert ep.episode_id >= split.episodes[ii - 1].episode_id


def test_get_uneven_splits():
    dataset = _construct_dataset(100)
    splits = dataset.get_uneven_splits(9)
    assert len(splits) == 9
    assert sum([len(split.episodes) for split in splits]) == 100
