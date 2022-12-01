#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import groupby, islice

import pytest

from habitat.core.dataset import Dataset, Episode
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal


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
    dataset: Dataset = Dataset()
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
    with pytest.raises(ValueError):
        splits = dataset.get_splits(10, 20)


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


@pytest.mark.parametrize(
    "num_episodes,num_splits",
    [(994, 64), (1023, 64), (1024, 64), (1025, 64), (10000, 9), (10000, 10)],
)
def test_get_splits_func(num_episodes: int, num_splits: int):
    dataset = _construct_dataset(num_episodes)
    splits = dataset.get_splits(num_splits, allow_uneven_splits=True)
    assert len(splits) == num_splits
    assert sum(len(split.episodes) for split in splits) == num_episodes
    splits = dataset.get_splits(num_splits, allow_uneven_splits=False)
    assert len(splits) == num_splits
    assert (
        sum(map(lambda s: s.num_episodes, splits))
        == (num_episodes // num_splits) * num_splits
    )


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


def test_iterator_scene_switching_episodes():
    total_ep = 1000
    max_repeat = 25
    dataset = _construct_dataset(total_ep)

    episode_iter = dataset.get_episode_iterator(
        max_scene_repeat_episodes=max_repeat, shuffle=False, cycle=True
    )
    episodes = sorted(dataset.episodes, key=lambda x: x.scene_id)

    for _ in range(max_repeat):
        episode = next(episode_iter)
        assert (
            episode.episode_id == episodes.pop(0).episode_id
        ), "episodes before max_repeat reached should be identical"

    episode = next(episode_iter)
    assert (
        episode.scene_id != episodes.pop(0).scene_id
    ), "After max_repeat episodes a scene switch doesn't happen."

    remaining_episodes = list(islice(episode_iter, total_ep - max_repeat - 1))
    assert len(remaining_episodes) == len(
        episodes
    ), "Remaining episodes should be identical."

    assert len({e.scene_id for e in remaining_episodes}) == len(
        set(map(lambda ep: ep.scene_id, remaining_episodes))
    ), "Next episodes should still include all scenes."

    cycled_episodes = list(islice(episode_iter, 4 * total_ep))
    assert (
        len(set(map(lambda x: x.episode_id, cycled_episodes))) == total_ep
    ), "Some episodes leaked after cycling."

    grouped_episodes = [
        list(g) for k, g in groupby(cycled_episodes, key=lambda x: x.scene_id)
    ]
    assert (
        len(sum(grouped_episodes, [])) == 4 * total_ep
    ), "Cycled episode iterator returned unexpected number of episodes."
    assert (
        len(grouped_episodes) == 4 * total_ep / max_repeat
    ), "The number of scene switches is unexpected."

    assert all(
        len(group) == max_repeat for group in grouped_episodes
    ), "Not all scene switches are equal to required number."


def test_iterator_scene_switching_episodes_without_shuffle_cycle():
    total_ep = 1000
    max_repeat = 25
    dataset = _construct_dataset(total_ep)
    episode_iter = dataset.get_episode_iterator(
        max_scene_repeat_episodes=max_repeat, shuffle=False, cycle=False
    )

    grouped_episodes = [
        list(g) for k, g in groupby(episode_iter, key=lambda x: x.scene_id)
    ]
    assert (
        len(sum(grouped_episodes, [])) == total_ep
    ), "The episode iterator returned unexpected number of episodes."
    assert (
        len(grouped_episodes) == total_ep / max_repeat
    ), "The number of scene switches is unexpected."

    assert all(
        len(group) == max_repeat for group in grouped_episodes
    ), "Not all scene stitches are equal to requirement."


def test_iterator_scene_switching_steps():
    total_ep = 1000
    max_repeat_steps = 250
    dataset = _construct_dataset(total_ep)

    episode_iter = dataset.get_episode_iterator(
        max_scene_repeat_steps=max_repeat_steps,
        shuffle=False,
        step_repetition_range=0.0,
    )
    episodes = sorted(dataset.episodes, key=lambda x: x.scene_id)

    episode = next(episode_iter)
    assert (
        episode.episode_id == episodes.pop(0).episode_id
    ), "After max_repeat_steps episodes a scene switch doesn't happen."

    # episodes before max_repeat reached should be identical
    for _ in range(max_repeat_steps):
        episode_iter.step_taken()

    episode = next(episode_iter)
    assert (
        episode.episode_id != episodes.pop(0).episode_id
    ), "After max_repeat_steps episodes a scene switch doesn't happen."

    remaining_episodes = list(islice(episode_iter, total_ep - 2))
    assert len(remaining_episodes) == len(
        episodes
    ), "Remaining episodes numbers aren't equal."

    assert len({e.scene_id for e in remaining_episodes}) == len(
        list(groupby(remaining_episodes, lambda ep: ep.scene_id))
    ), (
        "Next episodes should still be grouped by scene (before next "
        "switching)."
    )


def test_preserve_order():
    dataset = _construct_dataset(100)
    episodes = sorted(dataset.episodes, reverse=True, key=lambda x: x.scene_id)
    dataset.episodes = episodes[:]
    episode_iter = dataset.get_episode_iterator(shuffle=False, cycle=False)

    assert list(episode_iter) == episodes


def test_reset_goals():
    ep = NavigationEpisode(
        episode_id="0",
        scene_id="1",
        start_position=[0, 0, 0],
        start_rotation=[1, 0, 0, 0],
        goals=[NavigationGoal(position=[1, 2, 3])],
    )

    ep._shortest_path_cache = "Dummy"
    assert ep._shortest_path_cache is not None

    ep.goals = [NavigationGoal(position=[3, 4, 5])]
    assert ep._shortest_path_cache is None
