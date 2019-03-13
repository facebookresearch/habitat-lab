#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from habitat.tasks.nav.nav_task import NavigationEpisode, NavigationGoal
from habitat.utils.visualizations import maps
import scipy.misc


def example_pointnav_draw_target_birdseye_view():
    goal_radius = 0.5
    goal = NavigationGoal([10, 0.25, 10], goal_radius)
    agent_position = np.array([0, 0.25, 0])
    agent_rotation = -np.pi / 4

    dummy_episode = NavigationEpisode([goal], episode_id='dummy_id',
                                      scene_id='dummy_scene',
                                      start_position=agent_position,
                                      start_rotation=agent_rotation)
    target_image = maps.pointnav_draw_target_birdseye_view(
        agent_position, agent_rotation,
        np.asarray(dummy_episode.goals[0].position),
        goal_radius=dummy_episode.goals[0].radius)

    scipy.misc.imsave('pointnav_target_image.png', target_image)


def main():
    example_pointnav_draw_target_birdseye_view()


if __name__ == '__main__':
    main()
