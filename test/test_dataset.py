#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import groupby, islice

import pytest

from habitat.core.dataset import Dataset, Episode


def _construct_dataset(num_episodes, num_groups=10):
    episodes = []
    for i in range(num_episodes):
        episode = Episode(
            episode_id=str(i),
            scene_id="scene_id_" + str(i % num_groups),
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


def test_get_splits_num_episodes_specified():
    dataset = _construct_dataset(100)
    splits = dataset.get_splits(10, 3, False)
    assert len(splits) == 10
    for split in splits:
        assert len(split.episodes) == 3
    assert len(dataset.episodes) == 100

    dataset = _construct_dataset(100)
    splits = dataset.get_splits(10, 10)
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

    dataset = _construct_dataset(100)
    try:
        splits = dataset.get_splits(10, 20)
        assert False
    except AssertionError:
        pass


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
    dataset = _construct_dataset(10000)
    splits = dataset.get_splits(9, allow_uneven_splits=False)
    assert len(splits) == 9
    assert sum([len(split.episodes) for split in splits]) == (10000 // 9) * 9

    dataset = _construct_dataset(10000)
    splits = dataset.get_splits(9, allow_uneven_splits=True)
    assert len(splits) == 9
    assert sum([len(split.episodes) for split in splits]) == 10000

    dataset = _construct_dataset(10000)
    splits = dataset.get_splits(10, allow_uneven_splits=True)
    assert len(splits) == 10
    assert sum([len(split.episodes) for split in splits]) == 10000


def test_sample_episodes():
    dataset = _construct_dataset(1000)
    ep_iter = dataset.get_episode_iterator(
        num_episode_sample=1000, cycle=False
    )
    assert len(list(ep_iter)) == 1000

    ep_iter = dataset.get_episode_iterator(num_episode_sample=0, cycle=False)
    assert len(list(ep_iter)) == 0

    with pytest.raises(ValueError):
        dataset.get_episode_iterator(num_episode_sample=1001, cycle=False)

    ep_iter = dataset.get_episode_iterator(num_episode_sample=100, cycle=True)
    ep_id_list = [e.episode_id for e in list(islice(ep_iter, 100))]
    assert len(set(ep_id_list)) == 100
    next_episode = next(ep_iter)
    assert next_episode.episode_id in ep_id_list

    ep_iter = dataset.get_episode_iterator(num_episode_sample=0, cycle=False)
    with pytest.raises(StopIteration):
        next(ep_iter)


def test_iterator_cycle():
    dataset = _construct_dataset(100)
    ep_iter = dataset.get_episode_iterator(
        cycle=True, shuffle=False, group_by_scene=False
    )
    for i in range(200):
        episode = next(ep_iter)
        assert episode.episode_id == dataset.episodes[i % 100].episode_id

    ep_iter = dataset.get_episode_iterator(cycle=True, num_episode_sample=20)
    episodes = list(islice(ep_iter, 20))
    for i in range(200):
        episode = next(ep_iter)
        assert episode.episode_id == episodes[i % 20].episode_id


def test_iterator_shuffle():
    dataset = _construct_dataset(100)
    episode_iter = dataset.get_episode_iterator(shuffle=True)
    first_round_episodes = list(islice(episode_iter, 100))
    second_round_episodes = list(islice(episode_iter, 100))

    # both rounds should have same episodes but in different order
    assert sorted(first_round_episodes) == sorted(second_round_episodes)
    assert first_round_episodes != second_round_episodes

    # both rounds should be grouped by scenes
    first_round_scene_groups = [
        k for k, g in groupby(first_round_episodes, key=lambda x: x.scene_id)
    ]
    second_round_scene_groups = [
        k for k, g in groupby(second_round_episodes, key=lambda x: x.scene_id)
    ]
    assert len(first_round_scene_groups) == len(second_round_scene_groups)
    assert len(first_round_scene_groups) == len(set(first_round_scene_groups))


def test_iterator_scene_switching():
    total_ep = 1000
    max_repeat = 25
    dataset = _construct_dataset(total_ep)

    episode_iter = dataset.get_episode_iterator(max_scene_repeat=max_repeat)
    episodes = sorted(dataset.episodes, key=lambda x: x.scene_id)

    # episodes before max_repeat reached should be identical
    for i in range(max_repeat):
        episode = next(episode_iter)
        assert episode.episode_id == episodes.pop(0).episode_id

    remaining_episodes = list(islice(episode_iter, total_ep - max_repeat))
    # remaining episodes should be same but in different order
    assert len(remaining_episodes) == len(episodes)
    assert remaining_episodes != episodes
    assert sorted(remaining_episodes) == sorted(episodes)

    # next episodes should still be grouped by scene (before next switching)
    assert len(set([e.scene_id for e in remaining_episodes[:max_repeat]])) == 1
