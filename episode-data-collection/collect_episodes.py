#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import shutil
import cv2
import numpy as np
import random
import pickle
import torch
import matplotlib.pyplot as plt

import utils
import agent

from PIL import Image

import habitat
from env import SimpleRLEnv
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video
from habitat.utils.geometry_utils import quaternion_to_list
from habitat.config import read_write

from utils import draw_top_down_map, load_embedding, animate_episode

MIN_STEPS_TO_SAVE = 10

IMAGE_DIR = os.path.join("examples", "images")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

def shortest_path_navigation(args):
    if args.eval_model is not None:
        model, transform = load_embedding(args.eval_model)
        model.eval()
        overrides = ["+habitat/task/measurements@habitat.task.measurements.top_down_map=top_down_map"]
    else:
        overrides = []

    config = habitat.get_config(
        config_path=os.path.join("benchmark/nav/", args.env_config),
        overrides=overrides
    )

    with read_write(config):
        config.habitat.dataset.split = args.split

    if args.every_view:
        with read_write(config):
            config.habitat.environment.max_episode_steps = 100000

    if args.describe_goal:
        describer = utils.ImageDescriber()

    random.seed(42)
    
    with SimpleRLEnv(config=config) as env:
        goal_radius = env.episodes[0].goals[0].radius
        if goal_radius is None:
            goal_radius = config.habitat.simulator.forward_step_size
        
        follower = ShortestPathFollower(
            env.habitat_env.sim, goal_radius, False
        )

        turn_angle = config.habitat.simulator.turn_angle

        if args.verbose:
            print("Environment creation successful")
        
        for episode in range(len(env.episodes)):
            observations, info = env.reset(return_info=True)
            current_scene = env.habitat_env.current_episode.scene_id

            if random.random() < args.sample:
                continue

            # save the top-down map image
            maps = []
            
            goal = env.habitat_env.current_episode.goals[0]
            goal_category = ' '.join(goal.object_category.split('_'))
            if hasattr(goal, "view_points"):
                max_dist, view_n = 0, 0
                for i, view in enumerate(env.habitat_env.current_episode.goals[0].view_points):
                    # choose the furthest one because the object of interest is much better visible on the goal image in that case
                    if max_dist < np.linalg.norm(np.array(view.agent_state.position) - np.array(goal.position)):
                        max_dist = np.linalg.norm(np.array(view.agent_state.position) - np.array(goal.position))
                        view_n = i

                env.habitat_env.current_episode.goals[0].position = env.habitat_env.current_episode.goals[0].view_points[view_n].agent_state.position
                env.habitat_env.current_episode.goals[0].rotation = env.habitat_env.current_episode.goals[0].view_points[view_n].agent_state.rotation
                follower = agent.ImageNavShortestPathFollower(
                    env.habitat_env.sim, 
                    goal_radius,
                    env.habitat_env.current_episode.goals[0].view_points[view_n].agent_state.position, 
                    env.habitat_env.current_episode.goals[0].view_points[view_n].agent_state.rotation, 
                    False, 
                    turn_angle
                )

                # Save the goal image
                original_state = env.habitat_env.sim.get_agent_state() # save current agent state
                env.habitat_env.sim.set_agent_state(env.habitat_env.current_episode.goals[0].view_points[view_n].agent_state.position, 
                        env.habitat_env.current_episode.goals[0].view_points[view_n].agent_state.rotation) # move the agent in the goal state
                obs = env.habitat_env.sim.get_sensor_observations() # get the observation at the goal
                goal_image = obs["rgb"][:, :, :3]
                env.habitat_env.sim.set_agent_state(original_state.position, original_state.rotation) # return the agent in the starting position
            elif hasattr(observations, "instance_imagegoal"):
                goal_image = observations['instance_imagegoal']
            else:
                goal_image = None

            goal_description = ""
            if goal_image is not None and args.describe_goal:
                goal_description = describer.describe_image(goal_image, goal_category)

            if args.verbose:
                print("Agent stepping around inside environment.")

            images, actions, distances, views = [], [], [], []
            while not env.habitat_env.episode_over:
                best_action = follower.get_next_action()
                if best_action is None:
                    break
                
                images.append(observations["rgb"])
                distances.append(observations["pointgoal_with_gps_compass"][0])
                actions.append(best_action)

                if args.eval_model is not None:
                    top_down_map = draw_top_down_map(info, observations['rgb'].shape[0])
                    maps.append(top_down_map)
                    
                if args.every_view and best_action == HabitatSimActions.move_forward:
                    original_state = env.habitat_env.sim.get_agent_state()  # Save current agent state
                    position_views = []
                    turn_angle = config.habitat.simulator.turn_angle  # For instance, 30 degrees
                    num_views = 360 // turn_angle

                    for _ in range(num_views):
                        # Compute the new agent state rotated by 'turn_angle'
                        new_state = utils.rotate_agent_state(original_state, turn_angle)
                        
                        # Update the simulation state (this does not count as a step in the usual sense)
                        env.habitat_env.sim.set_agent_state(new_state.position, new_state.rotation)
                        
                        # Get new observations; this call should trigger a re-render based on the updated state.
                        obs = env.habitat_env.sim.get_sensor_observations()
                        position_views.append(obs["rgb"][:, :, :3])
                        
                        # Update original_state so that rotations accumulate if that’s the desired behavior.
                        original_state = new_state

                    views.append(position_views)
                    # Restore the original state if required.
                    env.habitat_env.sim.set_agent_state(original_state.position, original_state.rotation)
                                            
                observations, reward, done, info = env.step(best_action)

                if args.verbose:
                    print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
                        observations["pointgoal_with_gps_compass"][0],
                        observations["pointgoal_with_gps_compass"][1]))
            
            assert len(images) == len(distances)
            assert len(images) == len(actions)
            if len(images) >= MIN_STEPS_TO_SAVE:
                if args.save_per_scene:
                    dirname = os.path.join(args.save_dir, current_scene.split("/")[-1].split(".")[0], f"%0{len(str(len(env.episodes)))}d" % episode)
                else:
                    dirname = os.path.join(args.save_dir, f"%0{len(str(len(env.episodes)))}d" % episode)
                
                if os.path.exists(dirname):
                    shutil.rmtree(dirname)
                os.makedirs(dirname)

                images_to_video(images, dirname, "trajectory")
                with open(f"{dirname}/distances.pkl", "wb") as f:
                    pickle.dump(distances, f)
                with open(f"{dirname}/actions.pkl", "wb") as f:
                    pickle.dump(actions, f)
                with open(f"{dirname}/goal_category.txt", "w") as f:
                    f.write(goal_category + '\n')
                    f.write(goal_description)

                if goal_image is not None:
                    fig = plt.figure(frameon=False)
                    plt.imshow(goal_image, aspect='auto')
                    plt.axis('off') # No axes, ticks, or labels

                    # Save exactly what's inside the figure, with no extra whitespace
                    plt.savefig(f"{dirname}/goal_image.png", bbox_inches='tight', pad_inches=0)
                    plt.close()

                if args.every_view:
                    with open(f"{dirname}/views.pkl", "wb") as f:
                        pickle.dump(views, f)

                if args.eval_model:
                    model, transform = load_embedding(args.eval_model)
                    model.eval()
                    batch = torch.stack([transform(Image.fromarray(f)) for f in images]).to('cuda')
                    if args.eval_model in ('vip', 'r3m'):
                        batch = batch * 255

                    with torch.no_grad():
                        if args.eval_model.startswith(('dist_decoder_conf')):
                            goal = batch[-1:].repeat(len(batch), 1, 1, 1)
                            pred_distances, conf = model(batch, goal)
                            pred_distances = pred_distances.cpu().numpy()
                            conf = conf.cpu().numpy()
                        elif args.eval_model.startswith(('one_scene_decoder', 'dist_decoder',
                                                    'one_scene_quasi', 'quasi')):
                            goal = batch[-1:].repeat(len(batch), 1, 1, 1)
                            pred_distances = model(batch, goal).cpu().numpy().squeeze()
                        elif args.eval_model.startswith('vint'):
                            context = getattr(model, 'context_size', None) \
                                    or getattr(model.module, 'context_size', None)
                            N = batch.shape[0]
                            obs_list, goal_list, past_list = [], [], []
                            final_frame = batch[-1].unsqueeze(0)

                            for i in range(N):
                                obs_list.append(batch[i])
                                goal_list.append(final_frame[0])

                                if i >= context:
                                    past = batch[i-context:i]
                                else:
                                    n_pad = context - i
                                    pads = batch[0:1].repeat(n_pad, 1, 1, 1)
                                    past = torch.cat([pads, batch[0:i]], dim=0) if i > 0 else pads
                                past_list.append(past)

                            obs_imgs  = torch.stack(obs_list,  dim=0)
                            goal_imgs = torch.stack(goal_list, dim=0)
                            past_imgs = torch.stack(past_list, dim=0)

                            pred_distances = model(obs_imgs, goal_imgs, past_imgs)
                            pred_distances = pred_distances.cpu().numpy()
                        else:
                            emb = model(batch).cpu().numpy()
                            goal = emb[-1]
                            pred_distances = np.linalg.norm(emb - goal, axis=1)

                    
                    out_gif = os.path.join(dirname, f"model_eval.gif")
                    animate_episode(
                        frames=images,
                        maps=maps,
                        geo_distances=distances,
                        pred_distances=pred_distances,
                        goal_frame=images[-1],
                        out_video=os.path.join(dirname, "model_eval.mp4"),
                        max_len=args.max_len,
                        fps=10,
                    )
                    if args.verbose:
                        print(f"Saved 4-panel animation to {out_gif}")


            if args.verbose:
                print("Episode finished")

def main(args):
    shortest_path_navigation(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--env-config", type=str, default="instance_imagenav/instance_imagenav_hm3d_v2.yaml")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--every-view", action="store_true", default=False)
    parser.add_argument("--sample", type=float, default=0.0)
    parser.add_argument("--save-per-scene", action="store_true", default=False)
    parser.add_argument("--describe-goal", action="store_true", default=False)
    parser.add_argument("--eval-model", type=str, default=None)
    parser.add_argument("--max-len", type=int, default=100, help="Max frames to animate")
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    main(args)
