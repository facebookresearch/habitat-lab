#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""This script is used to generate customized Gibson training splits for the
PointNav task. The scenes in Gibson are ranked from 1-5 based on the
reconstruction quality (see https://arxiv.org/pdf/1904.01201.pdf for more
details). This script generates data for all scenes with a minimum quality of
q_thresh.
"""
import glob
import gzip
import json
import multiprocessing
import os
from os import path as osp

import tqdm

import habitat
from habitat.config.default import get_agent_config
from habitat.datasets.pointnav.pointnav_generator import (
    generate_pointnav_episode,
)

NUM_EPISODES_PER_SCENE = int(1e4)
# Sample all scenes with a minimum quality
QUAL_THRESH = 2


def safe_mkdir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass


def _generate_fn(scene):
    cfg = habitat.get_config(
        "benchmark/nav/pointnav/pointnav_habitat_test.yaml"
    )
    with habitat.config.read_write(cfg):
        cfg.habitat.simulator.scene = scene
        agent_config = get_agent_config(cfg.habitat.simulator)
        agent_config.sim_sensors.clear()

    sim = habitat.sims.make_sim("Sim-v0", config=cfg.habitat.simulator)

    dset = habitat.datasets.make_dataset("PointNav-v1")
    dset.episodes = list(
        generate_pointnav_episode(
            sim, NUM_EPISODES_PER_SCENE, is_gen_shortest_path=False
        )
    )
    for ep in dset.episodes:
        ep.scene_id = ep.scene_id[len("./data/scene_datasets/") :]

    scene_key = scene.split("/")[-1].split(".")[0]
    out_file = (
        f"./data/datasets/pointnav/gibson/v2/train_large/content/"
        f"{scene_key}.json.gz"
    )
    os.makedirs(osp.dirname(out_file), exist_ok=True)
    with gzip.open(out_file, "wt") as f:
        f.write(dset.to_json())


def generate_gibson_large_dataset():
    # Load train / val statistics
    with open(
        osp.join(osp.dirname(__file__), "gibson_dset_with_qual.json"), "r"
    ) as f:
        dataset_statistics = json.load(f)

    gibson_large_scene_keys = []
    for k, v in dataset_statistics.items():
        qual = v["qual"]
        if (
            v["split_full+"] == "train"
            and qual is not None
            and qual >= QUAL_THRESH
        ):
            gibson_large_scene_keys.append(k)

    scenes = glob.glob("./data/scene_datasets/gibson/*.glb")
    # Filter out invalid scenes
    _fltr = lambda x: x.split("/")[-1].split(".")[0] in gibson_large_scene_keys
    scenes = list(filter(_fltr, scenes))
    print(f"Total number of training scenes: {len(scenes)}")

    safe_mkdir("./data/datasets/pointnav/gibson/v2/train_large")
    with multiprocessing.Pool(8) as pool, tqdm.tqdm(total=len(scenes)) as pbar:
        for _ in pool.imap_unordered(_generate_fn, scenes):
            pbar.update()  # type: ignore[attr-defined]

    path = "./data/datasets/pointnav/gibson/v2/train_large/train_large.json.gz"
    with gzip.open(path, "wt") as f:
        json.dump(dict(episodes=[]), f)


if __name__ == "__main__":
    generate_gibson_large_dataset()
