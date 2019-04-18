#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil

import imageio

import habitat
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

IMAGE_DIR = os.path.join("examples", "images")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)


def shortest_path_example(mode):
    config = habitat.get_config(config_file="tasks/pointnav.yaml")
    env = habitat.Env(config=config)
    goal_radius = env.episodes[0].goals[0].radius
    if goal_radius is None:
        goal_radius = config.SIMULATOR.FORWARD_STEP_SIZE
    follower = ShortestPathFollower(env.sim, goal_radius, False)
    follower.mode = mode

    print("Environment creation successful")
    for episode in range(3):
        observations = env.reset()
        dirname = os.path.join(
            IMAGE_DIR, "shortest_path_example", mode, "%02d" % episode
        )
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        os.makedirs(dirname)
        print("Agent stepping around inside environment.")
        count_steps = 0
        while not env.episode_over:
            best_action = follower.get_next_action(
                env.current_episode.goals[0].position
            )
            observations = env.step(best_action.value)
            count_steps += 1
            im = observations["rgb"]
            imageio.imsave(os.path.join(dirname, "%03d.jpg" % count_steps), im)
        print("Episode finished after {} steps.".format(count_steps))


def main():
    shortest_path_example("geodesic_path")
    shortest_path_example("greedy")


if __name__ == "__main__":
    main()
