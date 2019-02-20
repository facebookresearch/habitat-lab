#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import habitat


def example():
    env = habitat.Env(config=habitat.get_config())
    observations = env.reset()

    while not env.episode_over:
        observations = env.step(env.action_space.sample())


if __name__ == "__main__":
    example()
