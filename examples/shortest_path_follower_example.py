#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil

import cv2
import numpy as np

import habitat
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video

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


def draw_top_down_map(info, heading, output_size):
    top_down_map = maps.colorize_topdown_map(
        info["top_down_map"]["map"], info["top_down_map"]["fog_of_war_mask"]
    )
    original_map_size = top_down_map.shape[:2]
    map_scale = np.array(
        (1, original_map_size[1] * 1.0 / original_map_size[0])
    )
    new_map_size = np.round(output_size * map_scale).astype(np.int32)
    # OpenCV expects w, h but map size is in h, w
    top_down_map = cv2.resize(top_down_map, (new_map_size[1], new_map_size[0]))

    map_agent_pos = info["top_down_map"]["agent_map_coord"]
    map_agent_pos = np.round(
        map_agent_pos * new_map_size / original_map_size
    ).astype(np.int32)
    top_down_map = maps.draw_agent(
        top_down_map,
        map_agent_pos,
        heading - np.pi / 2,
        agent_radius_px=top_down_map.shape[0] / 40,
    )
    return top_down_map


def shortest_path_example(mode):
    config = habitat.get_config(config_paths="configs/tasks/pointnav.yaml")
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
            observations, reward, done, info = env.step(best_action)
            im = observations["rgb"]
            top_down_map = draw_top_down_map(
                info, observations["heading"], im.shape[0]
            )
            output_im = np.concatenate((im, top_down_map), axis=1)
            images.append(output_im)
        images_to_video(images, dirname, "trajectory")
        print("Episode finished")


def main():
    shortest_path_example("geodesic_path")
    shortest_path_example("greedy")


if __name__ == "__main__":
    main()
