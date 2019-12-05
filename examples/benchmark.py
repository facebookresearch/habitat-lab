#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions


class ForwardOnlyAgent(habitat.Agent):
    def reset(self):
        pass

    def act(self, observations):
        action = HabitatSimActions.MOVE_FORWARD
        return {"action": action}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-config", type=str, default="configs/tasks/pointnav.yaml"
    )
    args = parser.parse_args()

    agent = ForwardOnlyAgent()
    benchmark = habitat.Benchmark(args.task_config)
    metrics = benchmark.evaluate(agent, num_episodes=10)

    for k, v in metrics.items():
        print("{}: {:.3f}".format(k, v))


if __name__ == "__main__":
    main()
