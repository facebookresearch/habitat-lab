#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gym

import habitat.gym  # noqa: F401


def example():
    # Note: Use with for the example testing, doesn't need to be like this on the README

    with gym.make("HabitatRenderPick-v0") as env:
        print("Environment creation successful")
        observations = env.reset()  # noqa: F841

        print("Agent acting inside environment.")
        count_steps = 0
        terminal = False
        while not terminal:
            observations, reward, terminal, info = env.step(
                env.action_space.sample()
            )  # noqa: F841
            count_steps += 1
        print("Episode finished after {} steps.".format(count_steps))


if __name__ == "__main__":
    example()
