#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys
from collections import defaultdict

import habitat
from habitat.config.default import get_config
from habitat.core.benchmark import Benchmark
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat_baselines.agents.simple_agents import (
    ForwardOnlyAgent,
    GoalFollower,
    RandomAgent,
    RandomForwardAgent,
)


"""
RandomAgent:
    Takes random actions. If within the goal radius, takes action STOP.
ForwardOnlyAgent:
    Only takes action MOVE_FORWARD. If within the goal radius, takes action STOP.
RandomForwardAgent:
    If within the goal radius, takes action STOP. Else:
    P(MOVE_FORWARD) = 80%
    P(TURN_LEFT) = 10%
    P(TURN_RIGHT) = 10%
GoalFollower:
    Tries to take direct route to the goal and takes action STOP when within
    the goal radius. Turns left or right if |angle_to_goal| > 15 deg.
ShortestPathAgent:
    takes the geodesic shortest path to the goal. If within the goal radius,
    takes action STOP.
"""


def shortest_path_benchmark(env, mode):
    """
    Custom benchmark for the shortest path agent because it requires access
    to habitat_env during each episode.
    mode either 'geodesic_path' or 'greedy'
    """
    goal_radius = env.episodes[0].goals[0].radius

    follower = ShortestPathFollower(env.sim, goal_radius, False)
    follower.mode = mode

    for episode in range(len(env.episodes)):
        env.reset()

        if not env.sim.is_navigable(env.current_episode.goals[0].position):
            print("Goal is not navigable.")
        while not env.episode_over:
            best_action = follower.get_next_action(
                env.current_episode.goals[0].position
            )
            observations = env.step(best_action)

    agg_metrics: Dict = defaultdict(float)
    for m, v in env.get_metrics().items():
        agg_metrics[m] += v
    avg_metrics = {k: v / len(env.episodes) for k, v in agg_metrics.items()}
    return avg_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-config", type=str, default="configs/tasks/vln_r2r.yaml"
    )
    parser.add_argument("--sp-mode", type=str, default="geodesic_path")
    args = parser.parse_args()
    config_env = get_config(args.task_config)
    benchmark = Benchmark(args.task_config)

    for agent_name in [
        "RandomAgent",
        "ForwardOnlyAgent",
        "RandomForwardAgent",
        "GoalFollower",
    ]:
        agent = getattr(sys.modules[__name__], agent_name)
        agent = agent(
            config_env.TASK.SUCCESS_DISTANCE, config_env.TASK.GOAL_SENSOR_UUID
        )
        metrics = benchmark.evaluate(agent)

        print(f"Benchmark for agent {agent_name}:")
        for k, v in metrics.items():
            print("{}: {:.3f}".format(k, v))
        print("")

    sp_metrics = shortest_path_benchmark(benchmark._env, args.sp_mode)

    print(f"Benchmark for agent ShortestPathAgent:")
    for k, v in sp_metrics.items():
        print("{}: {:.3f}".format(k, v))


if __name__ == "__main__":
    main()
