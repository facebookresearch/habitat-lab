#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
