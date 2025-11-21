#!/usr/bin/env python3

import argparse
import os
import shutil
import numpy as np
import random
import pickle
import torch
import matplotlib.pyplot as plt
import zipfile
import copy
import json
import cv2

import utils
import agent

from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

import habitat
from env import SimpleRLEnv
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video
from habitat.utils.geometry_utils import quaternion_to_list
from habitat.config import read_write

from utils import load_embedding, se3_world_T_cam_plusZ, K_from_fov, get_goal_info, projection_success_ratio, sample_safe_pose_by_depth, transform_rgb_bgr

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def shortest_path_navigation(args):
    distance_model, img_transform = load_embedding(args.eval_model)
    distance_model.eval()

    config = habitat.get_config(
        config_path=os.path.join("benchmark/nav/", args.env_config),
        overrides=["+habitat/task/measurements@habitat.task.measurements.top_down_map=top_down_map"]
    )

    with read_write(config):
        config.habitat.dataset.split = args.split
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth = False
        config.habitat.environment.max_episode_steps = 20_000

    random.seed(42)

    with SimpleRLEnv(config=config) as env:
        turn_angle = config.habitat.simulator.turn_angle

        # depth sensor values
        depth_sensor = config.habitat.simulator.agents[config.habitat.simulator.agents_order[0]].sim_sensors.depth_sensor
        depth_img_hfov = depth_sensor.hfov
        depth_img_width = depth_sensor.width
        depth_img_height = depth_sensor.height
        K = K_from_fov(depth_img_width, depth_img_height, depth_img_hfov)

        visited_scenes = set()
        corr_in_in, total_in_in = 0, 0
        corr_out_out, total_out_out = 0, 0
        corr_in_out, total_in_out = 0, 0
        for episode in tqdm(range(len(env.episodes))):
            observations, _ = env.reset(return_info=True)
            current_scene = env.habitat_env.current_episode.scene_id

            if current_scene in visited_scenes:
                continue

            # get goal camera intrinsics
            goal_obs, goal_state = get_goal_info(env)
            goal_image = goal_obs["rgb"][:, :, :3]  # Ensure RGB format
            goal_depth = np.squeeze(goal_obs["depth"]).astype(np.float32)
            goal_sensor_state = goal_state.sensor_states['depth']
            world_T_cam_goal = se3_world_T_cam_plusZ(goal_sensor_state.position, goal_sensor_state.rotation)

            # visited_scenes.add(current_scene)

            sampled_points = {"in_room": [], "out_room": []}

            # scene_id = current_scene.split('/')[-1]
            # episode_path = f'/cluster/scratch/lmilikic/vld_acc/{scene_id}_{episode}/{type}'
            # os.makedirs(episode_path, exist_ok=True)
            # cv2.imwrite(f"{episode_path}/goal_{idx}.png", transform_rgb_bgr(goal_image))

            for idx in range(2_000):
                # sample next point and get new observations
                new_pos, _ = sample_safe_pose_by_depth(env.habitat_env.sim, min_forward_clearance_m=3.0, max_trials=100)
                env.step(HabitatSimActions.turn_left)   
                observations, _, _, info = env.step(HabitatSimActions.turn_right)

                # compute the success ratio in the current position
                curr_depth = np.squeeze(observations["depth"]).astype(np.float32)
                agent_state = env.habitat_env.sim.get_agent_state(0)
                cam_state = agent_state.sensor_states["depth"]
                world_T_cam_curr = se3_world_T_cam_plusZ(cam_state.position, cam_state.rotation)

                sr = projection_success_ratio(goal_depth, K, world_T_cam_goal,
                                              curr_depth, K, world_T_cam_curr, depth_thresh=0.1)
                sr = round(sr, 2)
                if sr > 0:
                    continue

                # compute the temporal distance
                with torch.no_grad():
                    transformed_imgs = torch.stack([img_transform(Image.fromarray(f)) for f in [observations['rgb'], goal_image]]).to('cuda')
                    if args.eval_model.startswith("dist_vld"):
                        temporal_dist, conf = distance_model(transformed_imgs[:1], goal_image=transformed_imgs[1:])
                    else:
                        temporal_dist, conf = distance_model(transformed_imgs[:1], transformed_imgs[1:])
                    temporal_dist = temporal_dist.cpu().numpy()[0]

                sampled_point = {"temporal_dist": temporal_dist, "geodesc_dist": info["distance_to_goal"]}

                type = "out"
                # check if agent is in the same room as goal or not; if at any rotation success ratio is more than 0, then yes, otherwise no
                original_state = copy.deepcopy(env.habitat_env.sim.get_agent_state())  # Save current agent state
                for angle in range(depth_img_hfov, 360, depth_img_hfov):
                    # Compute the new agent state rotated by 'angle'
                    new_state = utils.rotate_agent_state(original_state, angle)
                    
                    # Get the success ratio at rotated position
                    obs = env.habitat_env.sim.get_observations_at(new_state.position, new_state.rotation, keep_agent_at_new_pose=True)                        
                    curr_depth = np.squeeze(obs["depth"]).astype(np.float32)
                    agent_state = env.habitat_env.sim.get_agent_state(0)
                    cam_state = agent_state.sensor_states["depth"]
                    world_T_cam_curr = se3_world_T_cam_plusZ(cam_state.position, cam_state.rotation)
                    sr = projection_success_ratio(goal_depth, K, world_T_cam_goal,
                                              curr_depth, K, world_T_cam_curr, depth_thresh=0.1)
                    sr = round(sr, 2)

                    if sr > 0:
                        sampled_points['in_room'].append(sampled_point)
                        type = "in"
                        break
                else:
                    sampled_points['out_room'].append(sampled_point)

                
                # with open(f"{episode_path}/labels_{idx}.json", 'w') as file:
                #     data = {
                #         "temporal_dist": str(round(temporal_dist, 2)),
                #         "geodesc_dist": str(round(observations["pointgoal_with_gps_compass"][0], 2))
                #     }

                #     json.dump(data, file, indent=4)

                # cv2.imwrite(f"{episode_path}/obs_{idx}.png", transform_rgb_bgr(observations['rgb']))

                if len(sampled_points['out_room']) >= 10 and len(sampled_points['in_room']) >= 10:
                    break

            in_room = sampled_points['in_room']
            out_room = sampled_points['out_room']
            
            # compute accuracy
            corr, cnt = 0, 0
            for in_room_point in in_room:
                for out_room_point in out_room:
                    cnt += 1
                    if in_room_point['temporal_dist'] >= out_room_point['temporal_dist']:
                        corr += int(in_room_point['geodesc_dist'] >= out_room_point['geodesc_dist'])
                    else:
                        corr += int(in_room_point['geodesc_dist'] < out_room_point['geodesc_dist'])
            
            corr_in_out += corr
            total_in_out += cnt
            print(f"In-Out Acc: {corr_in_out / total_in_out * 100:.2f}%")

            corr, cnt = 0, 0
            for i in range(len(in_room)):
                for j in range(i + 1, len(in_room)):
                    cnt += 1
                    if in_room[i]['temporal_dist'] >= in_room[j]['temporal_dist']:
                        corr += int(in_room[i]['geodesc_dist'] >= in_room[j]['geodesc_dist'])
                    else:
                        corr += int(in_room[i]['geodesc_dist'] < in_room[j]['geodesc_dist'])

            corr_in_in += corr
            total_in_in += cnt
            print(f"In-In Acc: {corr_in_in / total_in_in * 100:.2f}%")

            corr, cnt = 0, 0
            for i in range(len(out_room)):
                for j in range(i + 1, len(out_room)):
                    cnt += 1
                    if out_room[i]['temporal_dist'] >= out_room[j]['temporal_dist']:
                        corr += int(out_room[i]['geodesc_dist'] >= out_room[j]['geodesc_dist'])
                    else:
                        corr += int(out_room[i]['geodesc_dist'] < out_room[j]['geodesc_dist'])

            corr_out_out += corr
            total_out_out += cnt
            print(f"Out-Out Acc: {corr_out_out / total_out_out * 100:.2f}%")
            
        print("*" * 100)
        print(f"In-Out Acc: {corr_in_out / total_in_out * 100:.2f}%")
        print(f"In-In Acc: {corr_in_in / total_in_in * 100:.2f}%")
        print(f"Out-Out Acc: {corr_out_out / total_out_out * 100:.2f}%")


def main(args):
    shortest_path_navigation(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", type=str, default="instance_imagenav/instance_imagenav_hm3d_v2.yaml")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--eval-model", type=str, default="dist_decoder_conf_100max")
    args = parser.parse_args()

    main(args)
