#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil

import numpy as np

import habitat
from habitat.core.utils import try_cv2_import
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
)

cv2 = try_cv2_import()

IMAGE_DIR = os.path.join("examples", "images")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)


class SimpleRLEnv(habitat.RLEnv):
    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


def shortest_path_example(mode):
    config = habitat.get_config(
        config_paths="configs/tasks/objectnav_mp3d.yaml"
    )
    config.defrost()
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.TASK.SENSORS.append("HEADING_SENSOR")
    config.freeze()
    env = SimpleRLEnv(config=config)
    goal_radius = env.episodes[0].goals[0].radius
    if goal_radius is None:
        goal_radius = config.SIMULATOR.FORWARD_STEP_SIZE
    follower = ShortestPathFollower(env.habitat_env.sim, goal_radius, False)
    follower.mode = mode

    print("Environment creation successful")
    for episode in range(3):
        env.reset()
        dirname = os.path.join(
            IMAGE_DIR, "shortest_path_example", mode, "%02d" % episode
        )
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        os.makedirs(dirname)
        print("Agent stepping around inside environment.")
        images = []
        while not env.habitat_env.episode_over:
            best_action = follower.get_next_action(
                env.habitat_env.current_episode.goals[0].position
            )
            if best_action is None:
                break

            observations, reward, done, info = env.step(best_action)
            im = observations["rgb"]

            images.append(observations_to_image(observations, info))
        images_to_video(images, dirname, "trajectory")
        print("Episode finished")


def main():
    shortest_path_example("geodesic_path")
    shortest_path_example("greedy")


if __name__ == "__main__":
    main()
