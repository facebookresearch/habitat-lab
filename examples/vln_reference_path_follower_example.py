#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
from typing import List

import numpy as np

import habitat
from examples.shortest_path_follower_example import (
    SimpleRLEnv,
    draw_top_down_map,
)
from habitat.config.default_structured_configs import (
    HeadingSensorConfig,
    TopDownMapMeasurementConfig,
)
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations.utils import (
    append_text_underneath_image,
    images_to_video,
)

IMAGE_DIR = os.path.join("examples", "images")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)


def save_map(observations, info, images):
    im = observations["rgb"]
    top_down_map = draw_top_down_map(info, im.shape[0])
    output_im = np.concatenate((im, top_down_map), axis=1)
    output_im = append_text_underneath_image(
        output_im, observations["instruction"]["text"]
    )
    images.append(output_im)


def reference_path_example(mode):
    """
    Saves a video of a shortest path follower agent navigating from a start
    position to a goal. Agent follows the ground truth reference path by
    navigating to intermediate viewpoints en route to goal.
    Args:
        mode: 'geodesic_path' or 'greedy'
    """
    config = habitat.get_config(
        config_path="benchmark/nav/vln_r2r.yaml",
        overrides=[
            "habitat.task.measurements.success.success_distance=0.1",
            "habitat.dataset.split=val_seen",
        ],
    )
    with habitat.config.read_write(config):
        config.habitat.task.measurements.update(
            {"top_down_map": TopDownMapMeasurementConfig()}
        )
        config.habitat.task.lab_sensors.update(
            {"heading_sensor": HeadingSensorConfig()}
        )
    with SimpleRLEnv(config=config) as env:
        follower = ShortestPathFollower(
            env.habitat_env.sim, goal_radius=0.5, return_one_hot=False
        )
        follower.mode = mode
        print("Environment creation successful")

        for episode in range(3):
            env.reset()
            episode_id = env.habitat_env.current_episode.episode_id
            print(
                f"Agent stepping around inside environment. Episode id: {episode_id}"
            )

            dirname = os.path.join(
                IMAGE_DIR, "vln_reference_path_example", mode, "%02d" % episode
            )
            if os.path.exists(dirname):
                shutil.rmtree(dirname)
            os.makedirs(dirname)

            images: List[np.ndarray] = []
            steps = 0
            reference_path = env.habitat_env.current_episode.reference_path + [
                env.habitat_env.current_episode.goals[0].position
            ]
            for point in reference_path:
                done = False
                while not done:
                    best_action = follower.get_next_action(point)
                    if (
                        best_action is None
                        or best_action == HabitatSimActions.stop
                    ):
                        done = True
                        continue
                    observations, reward, done, info = env.step(best_action)
                    save_map(observations, info, images)
                    steps += 1

            print(f"Navigated to goal in {steps} steps.")
            images_to_video(images, dirname, str(episode_id))


if __name__ == "__main__":
    reference_path_example("geodesic_path")
