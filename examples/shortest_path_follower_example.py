#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys

import cv2

import habitat
from habitat.sims.habitat_simulator import SIM_NAME_TO_ACTION
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower


def shortest_path_example():
    config = habitat.get_config(config_file="tasks/pointnav.yaml")
    env = habitat.Env(config=config)
    goal_radius = env.episodes[0].goals[0].radius
    if goal_radius is None:
        goal_radius = config.SIMULATOR.FORWARD_STEP_SIZE
    follower = ShortestPathFollower(env.sim, goal_radius, False)

    print("Environment creation successful")
    for episode in range(3):
        observations = env.reset()
        print("Agent stepping around inside environment.")
        count_steps = 0
        while not env.episode_over:
            best_action = follower.get_next_action(
                env.current_episode.goals[0].position
            )
            observations = env.step(SIM_NAME_TO_ACTION[best_action.value])
            count_steps += 1
            if "pytest" not in sys.modules:
                im = observations["rgb"][:, :, ::-1].copy()
                cv2.putText(
                    im,
                    "Action: " + best_action.value,
                    (0, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                )
                cv2.imshow("im", im)
                cv2.waitKey(1)
        print("Episode finished after {} steps.".format(count_steps))


if __name__ == "__main__":
    shortest_path_example()
