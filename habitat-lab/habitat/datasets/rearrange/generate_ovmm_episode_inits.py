import argparse
import time
import random

import habitat
from gym import spaces
from habitat_baselines.config.default import get_config
from tqdm import tqdm
from habitat.utils.geometry_utils import quaternion_from_coeff
from habitat.core.utils import DatasetFloatJSONEncoder
import json, gzip
import os
import os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument("--cfg-path", type=str, default="habitat-lab/habitat/config/benchmark/rearrange/cat_nav_to_obj.yaml")
parser.add_argument(
    "opts",
    default=None,
    nargs=argparse.REMAINDER,
    help="Modify config options from command line",
)
args = parser.parse_args()


def set_episode(env, episode_id):
    episode = [ep for ep in env.episodes if ep.episode_id == episode_id][0]
    env.current_episode = episode

config = get_config(args.cfg_path, args.opts)

episodes_list = []

# load the episodes to copy over the metadata
episodes = json.load(gzip.open(config.habitat.dataset.data_path))
episodes_list = episodes['episodes']
# create the episode id to episode mapping
episode_mapping = {episode['episode_id']: episode for episode in episodes_list}
new_episodes_list = []

with habitat.Env(config=config) as env:
    env.reset()
    for i in range(env.number_of_episodes):
        # this will spawn the agent in the scene
        observations = env.reset()
        curr_episode = episode_mapping[env.current_episode.episode_id]
        # fetch the translation
        translation = env.sim.robot.sim_obj.translation
        curr_episode["start_position"] = [translation.x, translation.y, translation.z]
        # fetch the rotation
        curr_quat = env.sim.robot.sim_obj.rotation
        curr_rotation = [
            curr_quat.vector.x,
            curr_quat.vector.y,
            curr_quat.vector.z,
            curr_quat.scalar,
        ]
        curr_episode["start_rotation"] = curr_rotation
        # add the newly created episode
        new_episodes_list.append(curr_episode)

# copy over the episodes while retaining all other metadata
episodes['episodes'] = new_episodes_list
episodes_json = DatasetFloatJSONEncoder().encode(episodes)
target_episodes_file = config.habitat.dataset.data_path

# save the episodes
target_episodes_file = target_episodes_file.replace('.json.gz', '-with_init_poses.json.gz')
with gzip.open(target_episodes_file, "wt") as f:
    f.write(episodes_json)

