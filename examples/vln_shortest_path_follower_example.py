#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import json
import cv2
import numpy as np
import argparse
import habitat
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video

IMAGE_DIR = os.path.join("examples", "images")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)


class SimpleRLEnv(habitat.RLEnv):
    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


def draw_top_down_map(info, heading, output_size):
    top_down_map = maps.colorize_topdown_map(
        info["top_down_map"]["map"], info["top_down_map"]["fog_of_war_mask"]
    )
    original_map_size = top_down_map.shape[:2]
    map_scale = np.array(
        (1, original_map_size[1] * 1.0 / original_map_size[0])
    )
    new_map_size = np.round(output_size * map_scale).astype(np.int32)
    # OpenCV expects w, h but map size is in h, w
    top_down_map = cv2.resize(top_down_map, (new_map_size[1], new_map_size[0]))

    map_agent_pos = info["top_down_map"]["agent_map_coord"]
    map_agent_pos = np.round(
        map_agent_pos * new_map_size / original_map_size
    ).astype(np.int32)
    top_down_map = maps.draw_agent(
        top_down_map,
        map_agent_pos,
        heading - np.pi / 2,
        agent_radius_px=top_down_map.shape[0] / 40,
    )
    return top_down_map


def save_map(observations, info, images):
    im = observations["rgb"]
    top_down_map = draw_top_down_map(
        info, observations["heading"], im.shape[0]
    )
    output_im = np.concatenate((im, top_down_map), axis=1)
    shape = output_im.shape
    color = (255, 0, 0) 
    org = (5, shape[0] - 10) 

    fontScale = 0.5
    thickness = 1
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    y0, dy = shape[0] - 80, 20
    for i, line in enumerate(observations["instruction"]["text"].split('.')):
        y = y0 + i*dy
        cv2.putText(output_im, line, (5, y), font, fontScale, color, thickness, cv2.LINE_AA)

    images.append(output_im)


def shortest_path_example(mode, all_episodes=False):
    """
    Saves a video of a shortest path follower agent navigating from a start
    position to a goal. Agent navigates to intermediate viewpoints on the way.
    Args:
        mode: 'geodesic_path' or 'greedy'
        all_episodes: if True, runs for every episode. otherwise, 5. 
    """
    config = habitat.get_config(config_paths="configs/tasks/vln_r2r.yaml")
    config.defrost()
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.TASK.SENSORS.append("HEADING_SENSOR")
    config.freeze()
    env = SimpleRLEnv(config=config)

    follower = ShortestPathFollower(env.habitat_env.sim, goal_radius=0.5, return_one_hot=False)
    follower.mode = mode
    print("Environment creation successful")

    dirname = os.path.join(IMAGE_DIR, "vln_path_follow")
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
        os.makedirs(dirname)

    episodes_range = len(env.episodes) if all_episodes else 1
    for episode in range(episodes_range):
        env.reset()
        episode_id = env.habitat_env.current_episode.episode_id
        print(f"Agent stepping around inside environment. Episode id: {episode_id}")

        images = []
        error = False
        steps = 0
        path = env.habitat_env.current_episode.path + [
            env.habitat_env.current_episode.goals[0].position
        ]
        for point in path:
            done = False
            while not done:
                best_action = follower.get_next_action(point)
                if best_action == None:
                    break
                observations, reward, done, info = env.step(best_action)
                save_map(observations, info, images)
                steps += 1

        print(f'Navigated to goal in {steps} steps.')
        images_to_video(images, dirname, str(episode_id))
        images = []


def load_r2r_from_folder(dataset_dir):
    with open(dataset_dir) as f:
        json_array = json.load(f)
    return json_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
        '''
        Run a particular episode from the R2R Dataset
        Example invocation:
          python r2r_runner.py -e episode_id -m mode
        The -e argument is required and specifies the path_id.
        The --m argument is required to point to the r2r dataset train, test, val_seen, val_unseen

        ''',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-e', '--episode_id', required=False, help='episode_id to load', default=None)
    parser.add_argument('-m', '--mode', required=False, help='Data mode', default=None)
    args = parser.parse_args()
 
    if args.mode != None and args.episode_id != None:
        episode_id = args.episode_id
        mode = args.mode
        path = "data/datasets/R2R/hb_R2R_"
        single = "_single"
        json_name = ".json"
        json_array = load_r2r_from_folder(path + mode + json_name)

        if episode_id == "all":
            print("Episode Id : All")
            with open(path + "train" + single + json_name, 'w') as outfile:
                json.dump(json_array, outfile, indent=4)
        else:
            jsonObject = {}
            print("Episode Id : " + str(episode_id))
            jarray = []
            for jsonData in json_array["episodes"]:
                if int(jsonData["episode_id"]) == int(episode_id):
                    jarray.append(jsonData)
                    jsonObject["episodes"] = jarray
                    break
            with open(path + "train" + single + json_name, 'w') as outfile:
                json.dump(jsonObject, outfile, indent=4)

    shortest_path_example("geodesic_path")
