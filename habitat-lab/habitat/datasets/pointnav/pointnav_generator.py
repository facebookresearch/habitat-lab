#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""A minimum radius of a plane that a point should be part of to be
considered  as a target or source location. Used to filter isolated points
that aren't part of a floor.
"""

from typing import Dict, Generator, List, Optional, Sequence, Tuple, Union

import numpy as np

from habitat.core.simulator import ShortestPathPoint
from habitat.datasets.utils import get_action_shortest_path
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal

try:
    from habitat_sim.errors import GreedyFollowerError
except ImportError:
    GreedyFollower = BaseException
try:
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
except ImportError:
    habitat_sim = BaseException
ISLAND_RADIUS_LIMIT = 1.5


def _ratio_sample_rate(ratio: float, ratio_threshold: float) -> float:
    r"""Sampling function for aggressive filtering of straight-line
    episodes with shortest path geodesic distance to Euclid distance ratio
    threshold.

    :param ratio: geodesic distance ratio to Euclid distance
    :param ratio_threshold: geodesic shortest path to Euclid
    distance ratio upper limit till aggressive sampling is applied.
    :return: value between 0.008 and 0.144 for ratio [1, 1.1]
    """
    assert ratio < ratio_threshold
    return 20 * (ratio - 0.98) ** 2


def is_compatible_episode(
    s: Sequence[float],
    t: Sequence[float],
    sim: "HabitatSim",
    near_dist: float,
    far_dist: float,
    geodesic_to_euclid_ratio: float,
) -> Union[Tuple[bool, float], Tuple[bool, int]]:
    euclid_dist = np.power(np.power(np.array(s) - np.array(t), 2).sum(0), 0.5)
    if np.abs(s[1] - t[1]) > 0.5:  # check height difference to assure s and
        #  t are from same floor
        return False, 0
    d_separation = sim.geodesic_distance(s, [t])
    if d_separation == np.inf:
        return False, 0
    if not near_dist <= d_separation <= far_dist:
        return False, 0
    distances_ratio = d_separation / euclid_dist
    if distances_ratio < geodesic_to_euclid_ratio and (
        np.random.rand()
        > _ratio_sample_rate(distances_ratio, geodesic_to_euclid_ratio)
    ):
        return False, 0
    if sim.island_radius(s) < ISLAND_RADIUS_LIMIT:
        return False, 0
    return True, d_separation


def _create_episode(
    episode_id: Union[int, str],
    scene_id: str,
    start_position: List[float],
    start_rotation: List[float],
    target_position: List[float],
    shortest_paths: Optional[List[List[ShortestPathPoint]]] = None,
    radius: Optional[float] = None,
    info: Optional[Dict[str, float]] = None,
) -> Optional[NavigationEpisode]:
    goals = [NavigationGoal(position=target_position, radius=radius)]
    return NavigationEpisode(
        episode_id=str(episode_id),
        goals=goals,
        scene_id=scene_id,
        start_position=start_position,
        start_rotation=start_rotation,
        shortest_paths=shortest_paths,
        info=info,
    )


def generate_pointnav_episode(
    sim: "HabitatSim",
    num_episodes: int = -1,
    is_gen_shortest_path: bool = True,
    shortest_path_success_distance: float = 0.2,
    shortest_path_max_steps: int = 500,
    closest_dist_limit: float = 1,
    furthest_dist_limit: float = 30,
    geodesic_to_euclid_min_ratio: float = 1.1,
    number_retries_per_target: int = 10,
) -> Generator[NavigationEpisode, None, None]:
    r"""Generator function that generates PointGoal navigation episodes.

    An episode is trivial if there is an obstacle-free, straight line between
    the start and goal positions. A good measure of the navigation
    complexity of an episode is the ratio of
    geodesic shortest path position to Euclidean distance between start and
    goal positions to the corresponding Euclidean distance.
    If the ratio is nearly 1, it indicates there are few obstacles, and the
    episode is easy; if the ratio is larger than 1, the
    episode is difficult because strategic navigation is required.
    To keep the navigation complexity of the precomputed episodes reasonably
    high, we perform aggressive rejection sampling for episodes with the above
    ratio falling in the range [1, 1.1].
    Following this, there is a significant decrease in the number of
    straight-line episodes.


    :param sim: simulator with loaded scene for generation.
    :param num_episodes: number of episodes needed to generate
    :param is_gen_shortest_path: option to generate shortest paths
    :param shortest_path_success_distance: success distance when agent should
    stop during shortest path generation
    :param shortest_path_max_steps maximum number of steps shortest path
    expected to be
    :param closest_dist_limit episode geodesic distance lowest limit
    :param furthest_dist_limit episode geodesic distance highest limit
    :param geodesic_to_euclid_min_ratio geodesic shortest path to Euclid
    distance ratio upper limit till aggressive sampling is applied.
    :return: navigation episode that satisfy specified distribution for
    currently loaded into simulator scene.
    """
    episode_count = 0
    while episode_count < num_episodes or num_episodes < 0:
        target_position = sim.sample_navigable_point()

        if sim.island_radius(target_position) < ISLAND_RADIUS_LIMIT:
            continue

        for _retry in range(number_retries_per_target):
            source_position = sim.sample_navigable_point()

            is_compatible, dist = is_compatible_episode(
                source_position,
                target_position,
                sim,
                near_dist=closest_dist_limit,
                far_dist=furthest_dist_limit,
                geodesic_to_euclid_ratio=geodesic_to_euclid_min_ratio,
            )
            if is_compatible:
                break
        if is_compatible:
            angle = np.random.uniform(0, 2 * np.pi)
            source_rotation = [0.0, np.sin(angle / 2), 0, np.cos(angle / 2)]

            shortest_paths = None
            if is_gen_shortest_path:
                try:
                    shortest_paths = [
                        get_action_shortest_path(
                            sim,
                            source_position=source_position,
                            source_rotation=source_rotation,
                            goal_position=target_position,
                            success_distance=shortest_path_success_distance,
                            max_episode_steps=shortest_path_max_steps,
                        )
                    ]
                # Throws an error when it can't find a path
                except GreedyFollowerError:
                    continue

            episode = _create_episode(
                episode_id=episode_count,
                scene_id=sim.habitat_config.scene,
                start_position=source_position,
                start_rotation=source_rotation,
                target_position=target_position,
                shortest_paths=shortest_paths,
                radius=shortest_path_success_distance,
                info={"geodesic_distance": dist},
            )

            episode_count += 1
            yield episode
