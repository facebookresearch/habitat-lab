#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from collections import defaultdict
from typing import Dict

import habitat
from habitat.config.default import get_config
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower


def reference_path_benchmark(config, num_episodes=None):
    """
    Custom benchmark for the reference path agent because it requires access
    to habitat_env during each episode. Agent follows the ground truth
    reference path by navigating to intermediate viewpoints en route to goal.
    Args:
        config: Config
        num_episodes: Count of episodes to evaluate on.
    """
    with habitat.Env(config=config) as env:
        if num_episodes is None:
            num_episodes = len(env.episodes)

        follower = ShortestPathFollower(
            env.sim, goal_radius=0.5, return_one_hot=False
        )
        follower.mode = "geodesic_path"

        agg_metrics: Dict = defaultdict(float)
        for _ in range(num_episodes):
            env.reset()

            for point in env.current_episode.reference_path:
                while not env.episode_over:
                    best_action = follower.get_next_action(point)
                    if best_action == None:
                        break
                    env.step(best_action)

            while not env.episode_over:
                best_action = follower.get_next_action(
                    env.current_episode.goals[0].position
                )
                if best_action == None:
                    best_action = HabitatSimActions.stop
                env.step(best_action)

            for m, v in env.get_metrics().items():
                agg_metrics[m] += v

    avg_metrics = {k: v / num_episodes for k, v in agg_metrics.items()}
    return avg_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-config",
        type=str,
        default="benchmark/nav/vln_r2r.yaml",
    )
    args = parser.parse_args()
    config = get_config(args.task_config)

    metrics = reference_path_benchmark(config, num_episodes=10)

    print("Benchmark for Reference Path Follower agent:")
    for k, v in metrics.items():
        print("{}: {:.3f}".format(k, v))


if __name__ == "__main__":
    main()
