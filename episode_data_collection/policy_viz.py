#!/usr/bin/env python3

import argparse
import os
import shutil
import numpy as np
import random
import pickle
import torch
import matplotlib.pyplot as plt
import cv2


from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video
from habitat.utils.geometry_utils import quaternion_to_list
from habitat.config import read_write

from episode_data_collection.env import SimpleRLEnv
import episode_data_collection.utils as utils
from episode_data_collection.utils import draw_top_down_map, load_embedding, transform_rgb_bgr, se3_world_T_cam_plusZ, K_from_fov, make_mini_plot
from episode_data_collection.agent import ImageNavShortestPathFollower, PPOAgent

from rl_distance_train import distance_policy, dataset, distance_policy_gt
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def env_navigation(args):
    overrides = ["+habitat/task/measurements@habitat.task.measurements.top_down_map=top_down_map"]
    config = habitat.get_config(
        config_path=os.path.join("benchmark/nav/", args.env_config),
        overrides=overrides
    )

    with read_write(config):
        config.habitat.dataset.split = args.split
        config.habitat.environment.max_episode_steps = 300

    distance_norm = config.habitat.task.lab_sensors.pointgoal_with_gps_compass_sensor.distance_norm

    if args.describe_goal:
        describer = utils.ImageDescriber()

    random.seed(42)
    os.makedirs(args.save_dir, exist_ok=True)
    
    agent = PPOAgent.from_config(args.policy_checkpoint)
    distance_model, img_transform = load_embedding(agent.get_config().habitat_baselines.rl.ddppo.encoder_backbone)
    distance_model.eval()

    with SimpleRLEnv(config=config) as env:
        goal_radius = env.episodes[0].goals[0].radius
        turn_angle = config.habitat.simulator.turn_angle
        if goal_radius is None:
            goal_radius = config.habitat.simulator.forward_step_size
        
        for episode in tqdm(range(len(env.episodes))):
            observations, info = env.reset(return_info=True)
            current_scene = env.habitat_env.current_episode.scene_id
            agent.reset()

            if random.random() < args.sample:
                continue

            goal = env.habitat_env.current_episode.goals[0]
            
            try:
                goal_category = ' '.join(goal.object_category.split('_'))
            except:
                goal_category = ''

            if 'instance_imagegoal' in observations:
                goal_image = observations['instance_imagegoal']
            elif "imagegoal" in observations:
                goal_image = observations['imagegoal']
            elif hasattr(goal, "view_points"):
                init_agent_state = env.habitat_env.sim.get_agent_state()
                init_agent_pos = np.array(init_agent_state.position)
                
                min_dist = float("inf")
                view_n = 0
                # choose the closest view for the goal
                for i, view in enumerate(goal.view_points):
                    view_pos = np.array(view.agent_state.position)
                    dist = np.linalg.norm(view_pos - init_agent_pos)
                    if dist < min_dist:
                        min_dist = dist
                        view_n = i

                env.habitat_env.current_episode.goals[0].position = env.habitat_env.current_episode.goals[0].view_points[view_n].agent_state.position
                env.habitat_env.current_episode.goals[0].rotation = env.habitat_env.current_episode.goals[0].view_points[view_n].agent_state.rotation

                # Save the goal image
                original_state = env.habitat_env.sim.get_agent_state() # save current agent state
                env.habitat_env.sim.set_agent_state(env.habitat_env.current_episode.goals[0].view_points[view_n].agent_state.position, 
                        env.habitat_env.current_episode.goals[0].view_points[view_n].agent_state.rotation) # move the agent in the goal state
                obs = env.habitat_env.sim.get_sensor_observations() # get the observation at the goal
                goal_image = obs["rgb"][:, :, :3]
                env.habitat_env.sim.set_agent_state(original_state.position, original_state.rotation) # return the agent in the starting position
            else:
                raise ValueError("No goal image found in observations or episode goals.")

            goal_description = ""
            if goal_image is not None and args.describe_goal:
                goal_description = describer.describe_image(goal_image, goal_category)

            images, gt_distances, temporal_distances = [], [], []
            while not env.habitat_env.episode_over:
                # copy observations to avoid modifying the original
                try:
                    agent_obs = {k: observations[k] for k in agent.get_observation_space().keys()}
                except KeyError as e:
                    raise KeyError(f"Observation space mismatch: {e}. Available observations: {observations.keys()}") from e

                action = agent.act(agent_obs)

                if action is None:
                    break
                
                # log the current state
                print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
                observations["pointgoal_with_gps_compass"][0] * distance_norm,
                observations["pointgoal_with_gps_compass"][1]))
                gt_distances.append(observations["pointgoal_with_gps_compass"][0] * distance_norm)

                with torch.no_grad():
                    transformed_imgs = torch.stack([img_transform(Image.fromarray(f)) for f in [observations['rgb'], goal_image]]).to('cuda')
                    temporal_dist, conf = distance_model(transformed_imgs[:1], transformed_imgs[1:])
                    temporal_dist = temporal_dist.cpu().numpy()[0]
                print(f"Predicted distance {temporal_dist:2f}")
                temporal_distances.append(temporal_dist)

                # collect current view and state
                plot_img = make_mini_plot(gt_distances, temporal_distances, size=observations['rgb'].shape[0])

                top_down_map = draw_top_down_map(info, observations['rgb'].shape[0])
                combined_image = cv2.hconcat([observations["rgb"], top_down_map, goal_image, plot_img.astype(np.uint8)])
                images.append(combined_image)
                                            
                observations, reward, done, info = env.step(action)

            episode_id = f"%0{len(str(len(env.episodes)))}d" % episode
            images_to_video(images, args.save_dir, f"trajectory_{episode_id}", verbose=False)

def main(args):
    env_navigation(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--policy_checkpoint", type=str, required=True)
    parser.add_argument("--env-config", type=str, default="imagenav/imagenav_gibson.yaml")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--sample", type=float, default=0.0)
    parser.add_argument("--describe-goal", action="store_true", default=False)

    args = parser.parse_args()

    main(args)
