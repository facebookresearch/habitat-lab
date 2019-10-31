#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import textwrap

import numpy as np

import habitat
from examples.shortest_path_follower_example import (
    SimpleRLEnv,
    draw_top_down_map,
)
from habitat.core.utils import try_cv2_import
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video

cv2 = try_cv2_import()

IMAGE_DIR = os.path.join("examples", "images")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)


def append_text_to_image(orig_img, text):
    h, w, c = orig_img.shape
    font_size = 0.5
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    blank_image = np.zeros(orig_img.shape, dtype=np.uint8)

    char_size = cv2.getTextSize(" ", font, font_size, font_thickness)[0]
    wrapped_text = textwrap.wrap(text, width=int(w / char_size[0]))

    y = 0
    for line in wrapped_text:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        x = 10
        cv2.putText(
            blank_image,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    text_image = blank_image[0 : y + 10, 0:w]
    final = np.concatenate((orig_img, text_image), axis=0)
    return final


def save_map(observations, info, images):
    im = observations["rgb"]
    top_down_map = draw_top_down_map(
        info, observations["heading"], im.shape[0]
    )
    output_im = np.concatenate((im, top_down_map), axis=1)
    observations["instruction"]["text"]
    output_im = append_text_to_image(
        output_im, observations["instruction"]["text"]
    )
    images.append(output_im)


def shortest_path_example(mode):
    """
    Saves a video of a shortest path follower agent navigating from a start
    position to a goal. Agent follows the ground truth path by navigating to
    intermediate viewpoints en route to goal.
    Args:
        mode: 'geodesic_path' or 'greedy'
    """
    config = habitat.get_config(
        config_paths="configs/test/habitat_r2r_vln_test.yaml"
    )
    config.defrost()
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.TASK.SENSORS.append("HEADING_SENSOR")
    config.freeze()
    env = SimpleRLEnv(config=config)

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
            IMAGE_DIR, "vln_shortest_path_example", mode, "%02d" % episode
        )
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        os.makedirs(dirname)

        images = []
        steps = 0
        path = env.habitat_env.current_episode.path + [
            env.habitat_env.current_episode.goals[0].position
        ]
        for point in path:
            done = False
            while not done:
                best_action = follower.get_next_action(point)
                if best_action == None:
                    break
                observations, reward, done, info = env.step(best_action)
                save_map(observations, info, images)
                steps += 1

        print(f"Navigated to goal in {steps} steps.")
        images_to_video(images, dirname, str(episode_id))
        images = []


if __name__ == "__main__":
    shortest_path_example("geodesic_path")
