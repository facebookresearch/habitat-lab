#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import habitat


def example():
    config = habitat.get_config()
    dataset = habitat.make_dataset(
        id_dataset=config.DATASET.TYPE, config=config.DATASET
    )
    env = habitat.Env(config=config, dataset=dataset)
    observations = env.reset()

    while not env.episode_over:
        # randomly move around inside the environment
        observations = env.step(env.action_space.sample())


if __name__ == "__main__":
    example()
