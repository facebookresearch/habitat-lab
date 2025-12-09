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
from habitat.config.default_structured_configs import (
    TopDownMapMeasurementConfig,
    PointGoalWithGPSCompassSensorConfig,
)
from episode_data_collection.env import SimpleRLEnv
import episode_data_collection.utils as utils
from episode_data_collection.utils import draw_top_down_map, load_embedding, transform_rgb_bgr, se3_world_T_cam_plusZ, K_from_fov, make_mini_plot, get_goal_info
from episode_data_collection.agent import ImageNavShortestPathFollower, PPOAgent

from rl_distance_train import distance_policy, dataset, distance_policy_gt, geometric_distance_policy
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def env_navigation(args):
    agent = PPOAgent.from_config(args.policy_checkpoint)
    distance_model_rep = agent.get_config().habitat_baselines.rl.ddppo.encoder_backbone
    distance_model, img_transform = load_embedding(distance_model_rep)
    distance_model.eval()

    if args.env_config is not None:
        config = habitat.get_config(
            config_path=os.path.join("benchmark/nav/", args.env_config)
        )
    else:
        config = agent.get_config()

    with read_write(config):
        config.habitat.dataset.split = args.split
        config.habitat.environment.max_episode_steps = 300
        # Add TopDownMap measurement if missing
        if "top_down_map" not in config.habitat.task.measurements:
            config.habitat.task.measurements["top_down_map"] = TopDownMapMeasurementConfig()

        # Add PointGoalWithGPSCompassSensor if missing
        if "pointgoal_with_gps_compass_sensor" not in config.habitat.task.lab_sensors:
            config.habitat.task.lab_sensors[
                "pointgoal_with_gps_compass_sensor"
            ] = PointGoalWithGPSCompassSensorConfig()

    try:
        distance_norm = config.habitat.task.lab_sensors.pointgoal_with_gps_compass_sensor.distance_norm
    except:
        distance_norm = 1.0

    if args.describe_goal:
        describer = utils.ImageDescriber()

    random.seed(42)
    os.makedirs(args.save_dir, exist_ok=True)
    
    with SimpleRLEnv(config=config) as env:
        goal_radius = env.episodes[0].goals[0].radius
        turn_angle = config.habitat.simulator.turn_angle
        if goal_radius is None:
            goal_radius = config.habitat.simulator.forward_step_size
        
        outcome = {'success': 0, 'time_out': 0, 'near_goal': 0, 'wrong_goal': 0}
        for episode in tqdm(range(len(env.episodes))):
            observations, info = env.reset(return_info=True)
            current_scene = env.habitat_env.current_episode.scene_id
            agent.reset()

            if random.random() < args.sample:
                continue

            if episode != 7:
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
                print("No goal image found in observations or episode goals.")
                goal_image = None

            goal_description = ""
            if goal_image is not None and args.describe_goal:
                goal_description = describer.describe_image(goal_image, goal_category)

            # episode id (used both for per-step images and final video)
            episode_id = f"%0{len(str(len(env.episodes)))}d" % episode

            images, gt_distances, temporal_distances, confs = [], [], [], []
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

                if goal_image is not None:
                    with torch.no_grad():
                        transformed_imgs = torch.stack([img_transform(Image.fromarray(f)) for f in [observations['rgb'], goal_image]]).to('cuda')
                        if distance_model_rep.startswith("dist_vld"):
                            temporal_dist, conf = distance_model(transformed_imgs[:1], goal_image=transformed_imgs[1:])
                        else:
                            temporal_dist, conf = distance_model(transformed_imgs[:1], transformed_imgs[1:])
                        temporal_dist = temporal_dist.cpu().numpy()[0]
                        conf = conf.cpu().numpy()[0]
                    print(f"Predicted distance {temporal_dist:2f}")
                    temporal_distances.append(temporal_dist)
                    confs.append(conf)
                else:
                    temporal_distances.append(0)

                # collect current view and state
                plot_img = make_mini_plot(temporal_distances, confs, size=observations['rgb'].shape[0])

                top_down_map = draw_top_down_map(info, observations['rgb'].shape[0])
                h, w, _ = observations["rgb"].shape
                def resize(img):
                    return cv2.resize(img, (w, h))

                rgb = resize(observations["rgb"])
                tdm = resize(top_down_map)
                goal = resize(goal_image) if goal_image is not None else np.zeros_like(rgb)
                plot = resize(plot_img.astype(np.uint8))

                step_dir = os.path.join(args.save_dir, episode_id)
                os.makedirs(step_dir, exist_ok=True)
                step_idx = len(images)
                step_path = os.path.join(step_dir, f"{step_idx}.pdf")

                fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=150)

                axes[0].imshow(rgb)
                axes[0].set_title("Observation", fontsize=16)
                axes[0].axis('off')

                axes[1].imshow(goal)
                axes[1].set_title("Goal Image", fontsize=16)
                axes[1].axis('off')

                axes[2].imshow(top_down_map)
                axes[2].set_title("Top-Down Map", fontsize=16)
                axes[2].axis('off')

                axes[3].imshow(plot)
                axes[3].set_title("VLD over Time", fontsize=16)
                axes[3].axis('off')

                plt.tight_layout()
                fig.savefig(step_path, bbox_inches="tight", pad_inches=0.1)
                plt.close(fig)

                def add_label(img, text):
                    labeled = img.copy()
                    cv2.putText(
                        labeled, text, (10, 30),              # top-left corner
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,        # font + scale
                        (0, 0, 0), 2, cv2.LINE_AA       # black text, bold
                    )
                    return labeled

                rgb = add_label(rgb, "Obs")
                goal = add_label(goal, "Goal Img")
                tdm = add_label(tdm, "Map")

                # Arrange into 2x2 grid
                top_row = cv2.hconcat([rgb, goal])
                bottom_row = cv2.hconcat([tdm, plot])
                combined_image = cv2.vconcat([top_row, bottom_row])
                images.append(combined_image)
                                            
                observations, reward, done, info = env.step(action)

            is_success = bool(info['success'])
            if is_success:
                suffix = 'success'
            elif len(temporal_distances) > 995:
                suffix = 'time_out'
            elif info['distance_to_goal'] < 4 and temporal_distances[-1] < 5:
                suffix = 'near_goal'
            else:
                suffix = 'wrong_goal'

            outcome[suffix] += 1

            if not is_success:
                suffix += "_failure"

            fps = 10 if is_success else 50
            images_to_video(images, args.save_dir, f"trajectory_{episode_id}_{suffix}", verbose=False, fps=fps)

        total = outcome['success'] + outcome['time_out'] + outcome['near_goal'] + outcome['wrong_goal']
        print(f"Success: {outcome['success']}/{total} = {outcome['success'] / total * 100:.2f}%")
        print(f"Time Out Failure: {outcome['time_out']}/{total} = {outcome['time_out'] / total * 100:.2f}%")
        print(f"Near Goal Failure: {outcome['near_goal']}/{total} = {outcome['near_goal'] / total * 100:.2f}%")
        print(f"Wrong Goal Failure: {outcome['wrong_goal']}/{total} = {outcome['wrong_goal'] / total * 100:.2f}%")



def main(args):
    env_navigation(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--policy_checkpoint", type=str, required=True, help="Path to trained distance policy checkpoint.")
    parser.add_argument("--env-config", type=str, default=None, help="Path to environment configuration file.")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to use (train/val/test).")
    parser.add_argument("--sample", type=float, default=0.0, help="Fraction of episodes to sample.")
    parser.add_argument("--describe-goal", action="store_true", default=False, help="Whether to generate descriptions for goal images.")
    args = parser.parse_args()

    main(args)
