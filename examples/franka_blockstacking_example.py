#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import cv2
import numpy as np

import habitat
from habitat.core.logging import logger

SENSOR_KEY = "rgb"
CONFIG_FILE = "configs/tasks/franka_point.yaml"
BLOCK_CONFIGS_PATH = "data/objects/5boxes"
ROOM_SIZE = 5


def display_grayscale(image):
    img_bgr = np.repeat(image, 3, 2)
    cv2.imshow("Depth Sensor", img_bgr)
    cv2.waitKey(0)


def display_rgb(image):
    img_bgr = image[..., ::-1]
    cv2.imshow("RGB", img_bgr)
    cv2.waitKey(0)


def example(args):
    # Note: Use with for the example testing, doesn't need to be like this on the README

    with habitat.Env(config=habitat.get_config(CONFIG_FILE)) as env:
        logger.info("Environment creation successful")
        obj_templates_mgr = env.sim.get_object_template_manager()
        rigid_obj_mgr = env.sim.get_rigid_object_manager()

        block_template_ids = obj_templates_mgr.load_configs(BLOCK_CONFIGS_PATH)
        for tid in block_template_ids:
            obj = rigid_obj_mgr.add_object_by_template_id(tid)
            obj.translation = list(
                np.random.random(
                    3,
                )
                * ROOM_SIZE
            )
        logger.info("Blocks added to environment")

        observations = env.reset()  # noqa: F841
        if not args.no_render:
            display_rgb(observations[SENSOR_KEY])

        logger.info("Agent acting inside environment.")
        count_steps = 0
        logger.info(env.action_space)
        while not env.episode_over:
            action = env.action_space.sample()
            logger.info(f"Executing action: {action}")
            observations = env.step(action)  # noqa: F841
            if not args.no_render:
                display_rgb(observations[SENSOR_KEY])
            count_steps += 1
        logger.info("Episode finished after {} steps.".format(count_steps))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-render", action="store_true", default=False)
    args = parser.parse_args()
    example(args)
