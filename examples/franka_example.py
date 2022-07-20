#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import numpy as np

import habitat

SENSOR_KEY = "rgb"


def display_grayscale(image):
    img_bgr = np.repeat(image, 3, 2)
    cv2.imshow("Depth Sensor", img_bgr)
    _ = cv2.waitKey(0)


def display_rgb(image):
    img_bgr = image[..., ::-1]
    cv2.imshow("RGB", img_bgr)
    _ = cv2.waitKey(0)


def example():
    # Note: Use with for the example testing, doesn't need to be like this on the README

    with habitat.Env(
        config=habitat.get_config("configs/tasks/franka_point.yaml")
    ) as env:
        print("Environment creation successful")
        observations = env.reset()  # noqa: F841
        display_rgb(observations[SENSOR_KEY])

        print("Agent acting inside environment.")
        count_steps = 0
        print(env.action_space)
        while not env.episode_over:
            action = env.action_space.sample()
            print("Executing action:", action)
            observations = env.step(action)  # noqa: F841
            display_rgb(observations[SENSOR_KEY])
            count_steps += 1
        print("Episode finished after {} steps.".format(count_steps))


if __name__ == "__main__":
    example()
