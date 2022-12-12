#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
from typing import Any, Dict, Union

import imageio.v2 as imageio
import numpy as np

import habitat
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.core.agent import Agent
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.render_wrapper import overlay_frame
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
)

IMAGE_DIR = os.path.join("examples", "images")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)


class ShortestPathFollowerAgent(Agent):
    def __init__(self, env, goal_radius):
        self.env = env
        self.shortest_path_follower = ShortestPathFollower(
            sim=env.sim, goal_radius=goal_radius, return_one_hot=False
        )

    def act(self, observations) -> Union[int, str, Dict[str, Any]]:
        return self.shortest_path_follower.get_next_action(
            self.env.current_episode.goals[0].position
        )

    def reset(self) -> None:
        pass


def example_pointnav_draw_target_birdseye_view():
    goal_radius = 0.5
    goal = NavigationGoal(position=[10, 0.25, 10], radius=goal_radius)
    agent_position = [0, 0.25, 0]
    agent_rotation = -np.pi / 4

    dummy_episode = NavigationEpisode(
        goals=[goal],
        episode_id="dummy_id",
        scene_id="dummy_scene",
        start_position=agent_position,
        start_rotation=agent_rotation,  # type: ignore[arg-type]
    )
    agent_position = np.array(agent_position)
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
            agent_position = [7.8 * x_edge, 0.25, 7.8 * y_edge]
            agent_rotation = np.pi / 2

            dummy_episode = NavigationEpisode(
                goals=[goal],
                episode_id="dummy_id",
                scene_id="dummy_scene",
                start_position=agent_position,
                start_rotation=agent_rotation,  # type: ignore[arg-type]
            )
            agent_position = np.array(agent_position)
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
    config = habitat.get_config(
        config_path="benchmark/nav/pointnav/pointnav_habitat_test.yaml"
    )
    dataset = habitat.make_dataset(
        id_dataset=config.habitat.dataset.type, config=config.habitat.dataset
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


def example_top_down_map_measure():
    config = habitat.get_config(
        config_path="benchmark/nav/pointnav/pointnav_habitat_test.yaml"
    )
    dataset = habitat.make_dataset(
        id_dataset=config.habitat.dataset.type, config=config.habitat.dataset
    )
    with habitat.config.read_write(config):
        config.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=90,
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )

    num_episodes = 1
    with habitat.Env(config=config, dataset=dataset) as env:
        agent = ShortestPathFollowerAgent(
            env=env,
            goal_radius=config.habitat.task.measurements.success.success_distance,
        )
        for _ in range(num_episodes):
            observations = env.reset()
            agent.reset()

            info = env.get_metrics()
            frame = observations_to_image(observations, info)
            info.pop("top_down_map")
            frame = overlay_frame(frame, info)
            vis_frames = [frame]

            while not env.episode_over:
                action = agent.act(observations)
                if action is None:
                    break

                observations = env.step(action)
                info = env.get_metrics()
                frame = observations_to_image(observations, info)

                info.pop("top_down_map")
                frame = overlay_frame(frame, info)
                vis_frames.append(frame)

            current_episode = env.current_episode
            video_name = f"{os.path.basename(current_episode.scene_id)}_{current_episode.episode_id}"
            images_to_video(
                vis_frames, IMAGE_DIR, video_name, fps=6, quality=9
            )
            vis_frames.clear()


def main():
    example_pointnav_draw_target_birdseye_view()
    example_get_topdown_map()
    example_pointnav_draw_target_birdseye_view_agent_on_border()
    example_top_down_map_measure()


if __name__ == "__main__":
    main()
