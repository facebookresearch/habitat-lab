#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from examples.hitl.rearrange_v2.app_data import AppData
from examples.hitl.rearrange_v2.app_state_start_session import (
    AppStateStartSession,
)


def test_try_get_episode_indices():
    total_episode_count = 10
    data = AppData(max_user_count=2)

    # Canonical 1-user case.
    data.connected_users = {0: {"episodes": "1,7,9"}}
    assert AppStateStartSession._try_get_episode_indices(
        data, total_episode_count
    ) == [1, 7, 9]
    data.connected_users = {1: {"episodes": "3, 2, 0"}}
    assert AppStateStartSession._try_get_episode_indices(
        data, total_episode_count
    ) == [3, 2, 0]

    # Canonical 2-user case.
    data.connected_users = {
        0: {"episodes": "1,7,9"},
        1: {"episodes": "1,7,9"},
    }
    assert AppStateStartSession._try_get_episode_indices(
        data, total_episode_count
    ) == [1, 7, 9]

    # No user.
    data.connected_users = {}
    assert (
        AppStateStartSession._try_get_episode_indices(
            data, total_episode_count
        )
        == None
    )

    # Invalid formats.
    data.connected_users = {0: {"episodes": None}}
    assert (
        AppStateStartSession._try_get_episode_indices(
            data, total_episode_count
        )
        == None
    )
    data.connected_users = {0: {"episodes": 1}}
    assert (
        AppStateStartSession._try_get_episode_indices(
            data, total_episode_count
        )
        == None
    )
    data.connected_users = {0: {"episodes": 0.5}}
    assert (
        AppStateStartSession._try_get_episode_indices(
            data, total_episode_count
        )
        == None
    )
    data.connected_users = {0: {"episodes": {0: 2}}}
    assert (
        AppStateStartSession._try_get_episode_indices(
            data, total_episode_count
        )
        == None
    )

    # Empty list.
    data.connected_users = {0: {"episodes": []}}
    assert (
        AppStateStartSession._try_get_episode_indices(
            data, total_episode_count
        )
        == None
    )

    # Mismatching episodes.
    data.connected_users = {
        0: {"episodes": "1,7,9"},
        1: {"episodes": "3,2,0"},
    }
    assert (
        AppStateStartSession._try_get_episode_indices(
            data, total_episode_count
        )
        == None
    )

    # Out of range.
    data.connected_users = {0: {"episodes": [-1]}}
    assert (
        AppStateStartSession._try_get_episode_indices(
            data, total_episode_count
        )
        == None
    )
    data.connected_users = {0: {"episodes": [10]}}
    assert (
        AppStateStartSession._try_get_episode_indices(
            data, total_episode_count
        )
        == None
    )
    data.connected_users = {0: {"episodes": [0, 5, 11]}}
    assert (
        AppStateStartSession._try_get_episode_indices(
            data, total_episode_count
        )
        == None
    )
