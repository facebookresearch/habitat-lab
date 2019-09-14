#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil

import cv2
import numpy as np

import habitat
from habitat_baselines.agents.simple_agents import GoalFollower
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


def goal_follower_example():
    config = habitat.get_config(config_paths="configs/tasks/vln_r2r_mp3d.yaml")
    config.defrost()
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.TASK.SENSORS.append("HEADING_SENSOR")
    config.freeze()
    env = SimpleRLEnv(config=config)
    goal_radius = env.episodes[0].goals[0].radius

    if goal_radius is None:
        goal_radius = config.SIMULATOR.FORWARD_STEP_SIZE
    follower = GoalFollower(config.TASK.SUCCESS_DISTANCE, config.TASK.GOAL_SENSOR_UUID)

    print("Environment creation successful")
    for episode in range(len(env.episodes)):
        observations = env.reset()
        dirname = os.path.join(
            IMAGE_DIR, f"goal_follower_example"
        )
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        os.makedirs(dirname)
        print("Agent stepping around inside environment.")
        images = []

        print('Goal is navigable:', env.habitat_env.sim.is_navigable(env.habitat_env.current_episode.goals[0].position))
        while not env.habitat_env.episode_over:
            x = 0
            done = False
            i = 0
            while not done:
                action = follower.act(observations)
                observations, reward, done, info = env.step(action)
                im = observations["rgb"]
                top_down_map = draw_top_down_map(
                    info, observations["heading"], im.shape[0]
                )
                output_im = np.concatenate((im, top_down_map), axis=1)
                cv2.imwrite("examples/images/i.jpg", output_im)
                images.append(output_im)
                if i % 20 == 0:
                    print("Iteration : " + str(i))
                i += 1
        images_to_video(images, dirname, "trajectory_final")


if __name__ == "__main__":
    goal_follower_example()
