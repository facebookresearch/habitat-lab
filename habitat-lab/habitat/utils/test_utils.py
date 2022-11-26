#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from gym import spaces

from habitat.tasks.nav.nav import StopAction


def sample_non_stop_action(action_space, num_samples=1):
    samples = []
    for _ in range(num_samples):
        action = action_space.sample()
        while action["action"] == StopAction.name:
            action = action_space.sample()
        samples.append({"action": action})

    if num_samples == 1:
        return samples[0]["action"]
    else:
        return samples


def sample_non_stop_action_gym(action_space, num_samples=1):
    # Assuming action 0 is stopping
    assert isinstance(action_space, spaces.Discrete)
    samples = []
    for _ in range(num_samples):
        action = action_space.sample()
        while action == 0:
            action = action_space.sample()
        samples.append(action)

    if num_samples == 1:
        return samples[0]
    else:
        return samples
