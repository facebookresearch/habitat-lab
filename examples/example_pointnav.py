#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import habitat


def example():
    # Note: Use with for the example testing, doesn't need to be like this on the README

    with habitat.Env(
        config=habitat.get_config(
            "benchmark/nav/pointnav/pointnav_habitat_test.yaml"
        )
    ) as env:
        print("Environment creation successful")
        observations = env.reset()  # noqa: F841

        print("Agent stepping around inside environment.")
        count_steps = 0
        while not env.episode_over:
            observations = env.step(env.action_space.sample())  # noqa: F841
            count_steps += 1
        print("Episode finished after {} steps.".format(count_steps))


if __name__ == "__main__":
    example()
