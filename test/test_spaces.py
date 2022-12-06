#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gym

from habitat.core.spaces import ActionSpace, EmptySpace, ListSpace


def test_empty_space():
    space = EmptySpace()
    assert space.contains(space.sample())
    assert space.contains(None)
    assert not space.contains(0)


def test_action_space():
    space = ActionSpace(
        {
            "move": gym.spaces.Dict(
                {
                    "position": gym.spaces.Discrete(2),
                    "velocity": gym.spaces.Discrete(3),
                }
            ),
            "move_forward": EmptySpace(),
        }
    )
    assert space.contains(space.sample())
    assert space.contains(
        {"action": "move", "action_args": {"position": 0, "velocity": 1}}
    )
    assert space.contains({"action": "move_forward"})
    assert not space.contains([0, 1, 2])
    assert not space.contains({"zero": None})
    assert not space.contains({"action": "bad"})
    assert not space.contains({"action": "move"})
    assert not space.contains(
        {"action": "move", "action_args": {"position": 0}}
    )
    assert not space.contains(
        {"action": "move_forward", "action_args": {"position": 0}}
    )


def test_list_space():
    space = ListSpace(gym.spaces.Discrete(2), 5, 10)
    assert space.contains(space.sample())
    assert not space.contains(0)
    assert not space.contains([0] * 4)
    assert not space.contains([2] * 5)
    assert not space.contains([1] * 11)
