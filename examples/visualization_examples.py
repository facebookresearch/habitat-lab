#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os

import imageio
import numpy as np

import habitat
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.utils.visualizations import maps

IMAGE_DIR = os.path.join("examples", "images")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)


def example_pointnav_draw_target_birdseye_view():
    goal_radius = 0.5
    goal = NavigationGoal(position=[10, 0.25, 10], radius=goal_radius)
    agent_position = np.array([0, 0.25, 0])
    agent_rotation = -np.pi / 4

    dummy_episode = NavigationEpisode(
        goals=[goal],
        episode_id="dummy_id",
        scene_id="dummy_scene",
        start_position=agent_position,
        start_rotation=agent_rotation,
    )
    target_image = maps.pointnav_draw_target_birdseye_view(
        agent_position,
        agent_rotation,
        np.asarray(dummy_episode.goals[0].position),
        goal_radius=dummy_episode.goals[0].radius,
        agent_radius_px=25,
    )

    imageio.imsave(
        os.path.join(IMAGE_DIR, "pointnav_target_image.png"), target_image
    )


def example_pointnav_draw_target_birdseye_view_agent_on_border():
    goal_radius = 0.5
    goal = NavigationGoal(position=[0, 0.25, 0], radius=goal_radius)
    ii = 0
    for x_edge in [-1, 0, 1]:
        for y_edge in [-1, 0, 1]:
            if not np.bitwise_xor(x_edge == 0, y_edge == 0):
                continue
            ii += 1
            agent_position = np.array([7.8 * x_edge, 0.25, 7.8 * y_edge])
            agent_rotation = np.pi / 2

            dummy_episode = NavigationEpisode(
                goals=[goal],
                episode_id="dummy_id",
                scene_id="dummy_scene",
                start_position=agent_position,
                start_rotation=agent_rotation,
            )
            target_image = maps.pointnav_draw_target_birdseye_view(
                agent_position,
                agent_rotation,
                np.asarray(dummy_episode.goals[0].position),
                goal_radius=dummy_episode.goals[0].radius,
                agent_radius_px=25,
            )
            imageio.imsave(
                os.path.join(
                    IMAGE_DIR, "pointnav_target_image_edge_%d.png" % ii
                ),
                target_image,
            )


def example_get_topdown_map():
    config = habitat.get_config(config_paths="configs/tasks/pointnav.yaml")
    dataset = habitat.make_dataset(
        id_dataset=config.DATASET.TYPE, config=config.DATASET
    )
    with habitat.Env(config=config, dataset=dataset) as env:
        env.reset()
        top_down_map = maps.get_topdown_map_from_sim(
            env.sim, map_resolution=1024
        )
        recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        )
        top_down_map = recolor_map[top_down_map]
        imageio.imsave(
            os.path.join(IMAGE_DIR, "top_down_map.png"), top_down_map
        )


def main():
    example_pointnav_draw_target_birdseye_view()
    example_get_topdown_map()
    example_pointnav_draw_target_birdseye_view_agent_on_border()


if __name__ == "__main__":
    main()
