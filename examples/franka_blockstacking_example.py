#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import numpy as np
from display_utils import display_rgb

import habitat
from habitat.core.logging import logger

SENSOR_KEY = "rgb"
CONFIG_FILE = "benchmark/nav/pointnav/pointnav_franka.yaml"
BLOCK_LOC_RANGE = 2.0
BLOCK_LOC_CENTER = [-5.0, 0.1, 8.5]
BLOCK_SCALE = 0.2
NUM_BLOCKS = 6
UNIFORM_SCALE = 2
UNIFORM_OFFSET = UNIFORM_SCALE / 2


def example(render):
    # Note: Use with for the example testing, doesn't need to be like this on the README

    with habitat.Env(config=habitat.get_config(CONFIG_FILE)) as env:
        logger.info("Environment creation successful")

        obj_templates_mgr = env.sim.get_object_template_manager()
        rigid_obj_mgr = env.sim.get_rigid_object_manager()

        cube_handle = obj_templates_mgr.get_template_handles("cubeSolid")[0]
        cube_template_cpy = obj_templates_mgr.get_template_by_handle(
            cube_handle
        )
        cube_template_cpy.scale = np.array([BLOCK_SCALE] * 3)
        obj_templates_mgr.register_template(
            cube_template_cpy, "my_scaled_cube"
        )

        for _ in range(NUM_BLOCKS):
            obj = rigid_obj_mgr.add_object_by_template_handle("my_scaled_cube")

            # blocks placed around BLOCK_LOC_CENTER with uniform distribution and range BLOCK_LOC_RANGE
            offset = (
                np.random.random(3) * UNIFORM_SCALE - UNIFORM_OFFSET
            ) * BLOCK_LOC_RANGE
            offset[1] = 0  #  no change in z
            obj.translation = BLOCK_LOC_CENTER + offset

        logger.info("Blocks added to environment")

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
