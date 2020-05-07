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
import os.path as osp

import tqdm

import habitat
import habitat_sim
from habitat.datasets.pointnav.pointnav_generator import (
    generate_pointnav_episode,
)

num_episodes_per_scene = int(1e4)


def safe_mkdir(path):
    try:
        os.mkdir(path)
    except:
        pass


def _generate_fn(scene):
    cfg = habitat.get_config()
    cfg.defrost()
    cfg.SIMULATOR.SCENE = scene
    cfg.SIMULATOR.AGENT_0.SENSORS = []
    cfg.freeze()

    sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)

    dset = habitat.datasets.make_dataset("PointNav-v1")
    dset.episodes = list(
        generate_pointnav_episode(
            sim, num_episodes_per_scene, is_gen_shortest_path=False
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


# Load train / val statistics
dataset_statistics = json.load(
    open("./examples/gibson_large_generation/gibson_dset_with_qual.json")
)

# all Sample scenes with a minimum quality
q_thresh = 2

gibson_large_scene_keys = []
for k, v in dataset_statistics.items():
    qual = v["qual"]
    if v["split_full+"] == "train" and qual is not None and qual >= q_thresh:
        gibson_large_scene_keys.append(k)

scenes = glob.glob("./data/scene_datasets/gibson/*.glb")
# Filter out invalid scenes
_fltr = lambda x: x.split("/")[-1].split(".")[0] in gibson_large_scene_keys
scenes = list(filter(_fltr, scenes))
print(f"Total number of training scenes: {len(scenes)}")

safe_mkdir("./data/datasets/pointnav/gibson/v2/train_large")
with multiprocessing.Pool(8) as pool, tqdm.tqdm(total=len(scenes)) as pbar:
    for _ in pool.imap_unordered(_generate_fn, scenes):
        pbar.update()

path = "./data/datasets/pointnav/gibson/v2/train_large/train_large.json.gz"
with gzip.open(path, "wt") as f:
    json.dump(dict(episodes=[]), f)
