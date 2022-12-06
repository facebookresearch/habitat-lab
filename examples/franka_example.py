#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

from display_utils import display_rgb

import habitat
from habitat.core.logging import logger

SENSOR_KEY = "rgb"


def example(render):
    # Note: Use with for the example testing, doesn't need to be like this on the README

    with habitat.Env(
        config=habitat.get_config(
            "benchmark/nav/pointnav/pointnav_franka.yaml"
        )
    ) as env:
        logger.info("Environment creation successful")
        observations = env.reset()  # noqa: F841
        if render:
            display_rgb(observations[SENSOR_KEY])

        logger.info("Agent acting inside environment.")
        count_steps = 0
        logger.info(env.action_space)
        while not env.episode_over:
            action = env.action_space.sample()
            logger.info(f"Executing action: {action}")
            observations = env.step(action)  # noqa: F841
            if render:
                display_rgb(observations[SENSOR_KEY])
            count_steps += 1
        logger.info("Episode finished after {} steps.".format(count_steps))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-render", action="store_true", default=False)
    args = parser.parse_args()
    render = not args.no_render
    example(render=render)
