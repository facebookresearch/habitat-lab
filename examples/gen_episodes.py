#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import random
import sys

import numpy as np
from tqdm import tqdm

import habitat
import habitat.datasets.pointnav.pointnav_generator as pointnav_generator
from habitat.config.default import get_config
from habitat.sims import make_sim


def generate_pointnav_dataset(config, num_episodes):
    sim = make_sim(id_sim=config.SIMULATOR.TYPE, config=config.SIMULATOR)

    sim.seed(config.SEED)
    random.seed(config.SEED)
    generator = pointnav_generator.generate_pointnav_episode(
        sim=sim,
        shortest_path_success_distance=config.TASK.SUCCESS_DISTANCE,
        shortest_path_max_steps=config.ENVIRONMENT.MAX_EPISODE_STEPS,
    )
    episodes = []
    for i in range(num_episodes):
        print(f"Generating episode {i+1}/{num_episodes}")
        episode = next(generator)
        episodes.append(episode)

    dataset = habitat.Dataset()
    dataset.episodes = episodes
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate episodes for habitat"
    )
    parser.add_argument(
        "--task-config",
        default="configs/tasks/pointnav.yaml",
        help="Task configuration file for initializing a Habitat environment",
    )
    parser.add_argument("--scenes", help="Scenes")
    parser.add_argument(
        "-n",
        "--num_episodes",
        type=int,
        default=10,
        help="Number of episodes to generate",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="Output file for episodes",
    )
    args = parser.parse_args()
    opts = []
    config = habitat.get_config(args.task_config.split(","), opts)
    dataset_type = config.DATASET.TYPE
    if args.scenes is not None:
        config.defrost()
        config.SIMULATOR.SCENE = args.scenes
        config.freeze()
    print(config)
    dataset = None
    if dataset_type == "PointNav-v1":
        dataset = generate_pointnav_dataset(config, args.num_episodes)
    else:
        print(f"Unknown dataset type: {dataset_type}")
    if dataset is not None:
        args.output.write(dataset.to_json())
