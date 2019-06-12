#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import cv2
import imageio
import numpy as np
import os
import torch
import tqdm


import habitat
from habitat import logger
from torch.utils import tensorboard

from config.default import get_config as cfg_baseline
from habitat.config.default import get_config
from rl.ppo import PPO, Policy
from rl.ppo.utils import batch_obs
from train_ppo import make_env_fn
from habitat.utils.visualizations import maps


def images_to_video(images, output_dir, video_name):
    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
    video_path = os.path.join(output_dir, video_name)
    writer = imageio.get_writer(video_path, fps=10)
    for im in tqdm.tqdm(images):
        writer.append_data(im)
    writer.close()
    logger.info(
        "Generated video: {}".format(os.path.join(output_dir, video_name))
    )
    return video_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--sim-gpu-id", type=int, required=True)
    parser.add_argument("--pth-gpu-id", type=int, required=True)
    parser.add_argument("--num-processes", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--count-test-episodes", type=int, default=100)
    parser.add_argument(
        "--sensors",
        type=str,
        default="RGB_SENSOR,DEPTH_SENSOR",
        help="comma separated string containing different"
        "sensors to use, currently 'RGB_SENSOR' and"
        "'DEPTH_SENSOR' are supported",
    )
    parser.add_argument(
        "--task-config",
        type=str,
        default="configs/tasks/pointnav.yaml",
        help="path to config yaml containing information about task",
    )
    parser.add_argument("--video", type=int, default=0, choices=[0, 1], help="1 for add video")
    parser.add_argument("--out-dir-video", type=str, help="video dir")
    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.pth_gpu_id))

    writer_kwargs = dict(
        log_dir="tb_video", purge_step=0, flush_secs=30
    )

    env_configs = []
    baseline_configs = []


    for _ in range(args.num_processes):
        config_env = get_config(config_paths=args.task_config)
        config_env.defrost()
        config_env.DATASET.SPLIT = "val"

        agent_sensors = args.sensors.strip().split(",")
        for sensor in agent_sensors:
            assert sensor in ["RGB_SENSOR", "DEPTH_SENSOR"]
        config_env.SIMULATOR.AGENT_0.SENSORS = agent_sensors
        if args.video == 1:
            config_env.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config_env.TASK.MEASUREMENTS.append("COLLISIONS")
        config_env.freeze()
        env_configs.append(config_env)

        config_baseline = cfg_baseline()
        baseline_configs.append(config_baseline)

    assert len(baseline_configs) > 0, "empty list of datasets"

    envs = habitat.VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(
                zip(env_configs, baseline_configs, range(args.num_processes))
            )
        ),
    )

    ckpt = torch.load(args.model_path, map_location=device)

    actor_critic = Policy(
        observation_space=envs.observation_spaces[0],
        action_space=envs.action_spaces[0],
        hidden_size=512,
        goal_sensor_uuid=env_configs[0].TASK.GOAL_SENSOR_UUID,
    )
    actor_critic.to(device)

    ppo = PPO(
        actor_critic=actor_critic,
        clip_param=0.1,
        ppo_epoch=4,
        num_mini_batch=32,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        lr=2.5e-4,
        eps=1e-5,
        max_grad_norm=0.5,
    )

    ppo.load_state_dict(ckpt["state_dict"])

    actor_critic = ppo.actor_critic

    observations = envs.reset()
    batch = batch_obs(observations)
    for sensor in batch:
        batch[sensor] = batch[sensor].to(device)

    episode_rewards = torch.zeros(envs.num_envs, 1, device=device)
    episode_spls = torch.zeros(envs.num_envs, 1, device=device)
    episode_success = torch.zeros(envs.num_envs, 1, device=device)
    episode_counts = torch.zeros(envs.num_envs, 1, device=device)
    current_episode_reward = torch.zeros(envs.num_envs, 1, device=device)

    test_recurrent_hidden_states = torch.zeros(
        args.num_processes, args.hidden_size, device=device
    )
    not_done_masks = torch.zeros(args.num_processes, 1, device=device)
    stats_episodes = set()
    if args.video == 1:
        rgb_frames = [[]] * args.num_processes
        if not os.path.exists(args.out_dir_video):
            os.makedirs(args.out_dir_video)
    else:
        rgb_frames = None

    with (tensorboard.SummaryWriter(**writer_kwargs)) as writer:
        while episode_counts.sum() < args.count_test_episodes:
            current_episodes = envs.current_episodes()


            with torch.no_grad():
                _, actions, _, test_recurrent_hidden_states = actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    not_done_masks,
                    deterministic=False,
                )

            outputs = envs.step([a[0].item() for a in actions])

            observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
            batch = batch_obs(observations)
            for sensor in batch:
                batch[sensor] = batch[sensor].to(device)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=device,
            )

            for i in range(not_done_masks.shape[0]):
                if not_done_masks[i].item() == 0:
                    episode_spls[i] += infos[i]["spl"]
                    if infos[i]["spl"] > 0:
                        episode_success[i] += 1

            rewards = torch.tensor(
                rewards, dtype=torch.float, device=device
            ).unsqueeze(1)
            current_episode_reward += rewards
            episode_rewards += (1 - not_done_masks) * current_episode_reward
            episode_counts += 1 - not_done_masks
            current_episode_reward *= not_done_masks

            # added for video support
            next_episodes = envs.current_episodes()
            envs_to_pause = []
            n_envs = envs.num_envs
            for i in range(n_envs):
                if next_episodes[i].episode_id in stats_episodes:
                    envs_to_pause.append(i)

                if not_done_masks[i].item() == 0:
                    # new episode ended, record stats
                    stats_episodes.add(current_episodes[i].episode_id)
                    if args.video == 1:

                        video_name = "{}_{}_{}_{:.2f}".format(
                            current_episodes[i].episode_id,
                            "apt",
                            "spl",
                            infos[i]["spl"],
                        )

                        video_path = images_to_video(
                            rgb_frames[i],
                            args.out_dir_video,
                            video_name,
                        )

                        frame_tensors = [torch.from_numpy(np_arr).unsqueeze(0) for np_arr in rgb_frames[i]]
                        video_tensor = torch.cat(tuple(frame_tensors))
                        video_tensor = video_tensor.permute(0, 3, 1, 2).unsqueeze(0)  # Shape (N,T,C,H,W)
                        writer.add_video(video_path, video_tensor, fps=8)
                        rgb_frames[i] = []

                elif args.video == 1:
                    observation_size = observations[i]["rgb"].shape[0]
                    egocentric_view = observations[i]["rgb"][:, :, :3]
                    # print("=============observations {}".format(observations[i]["depth"].shape))
                    egocentric_depth = (observations[i]["depth"].squeeze() * 255).astype(np.uint8)
                    egocentric_depth = np.stack([egocentric_depth for _ in range(3)], axis=2)
                    egocentric_depth = cv2.applyColorMap(egocentric_depth, cv2.COLORMAP_PINK)
                    # draw collision
                    if infos[i]["collisions"]["is_collision"]:
                        mask = np.ones(
                            (egocentric_view.shape[0], egocentric_view.shape[1])
                        )
                        mask[30:-30, 30:-30] = 0
                        mask = mask == 1
                        alpha = 0.5
                        egocentric_view[mask] = (
                            alpha * np.array([255, 0, 0])
                            + (1.0 - alpha) * egocentric_view
                        )[mask]

                    top_down_map = infos[i]["top_down_map"]["map"]
                    top_down_map = maps.colorize_topdown_map(top_down_map)
                    map_agent_pos = infos[i]["top_down_map"][
                        "agent_map_coord"
                    ]

                    top_down_map = maps.draw_agent(
                        image=top_down_map,
                        agent_center_coord=map_agent_pos,
                        agent_rotation=infos[i]["top_down_map"]["agent_angle"],
                        agent_radius_px=8,
                    )

                    if top_down_map.shape[0] > top_down_map.shape[1]:
                        top_down_map = np.rot90(top_down_map, 1)

                    # print("old shape {}".format(top_down_map.shape))
                    old_h, old_w, _ = top_down_map.shape
                    top_down_height = observation_size
                    top_down_width = int(float(top_down_height)/old_h * old_w)
                    # print("new height {}, new width {}".format(top_down_height, top_down_width))

                    # cv2 resize dsize is width first
                    top_down_map = cv2.resize(
                                                top_down_map,
                                                (top_down_width, top_down_height),
                                                interpolation=cv2.INTER_CUBIC,
                                                )
                    frame = np.concatenate((egocentric_view, egocentric_depth, top_down_map), axis=1)

                    # print("ego shape {}, type {}".format(egocentric_view.shape, egocentric_view.dtype))
                    # print("topdown shape {}, type {}".format(top_down_map.shape, top_down_map.dtype))
                    # print("final frame shape: {} ".format(frame.shape))


                    # # no zooming top_down_map
                    # top_down_frame = np.full_like(egocentric_view, 255)
                    # top_down_frame[: top_down_map.shape[0], :top_down_map.shape[1], ] = top_down_map
                    # frame = np.concatenate((egocentric_view, top_down_frame), axis=1)

                    # make frame size divisible by 16 to accommodate imageio default basic_block_size
                    if frame.shape[1] % 16 != 0:
                        white_strip = np.full((frame.shape[0], 16-frame.shape[1]%16, frame.shape[2]), 255, dtype=np.uint8)
                        frame = np.concatenate((frame, white_strip), axis=1)

                    rgb_frames[i].append(frame)

            if len(envs_to_pause) > 0:
                state_index = list(range(envs.num_envs))

                for idx in reversed(envs_to_pause):
                    state_index.pop(idx)
                    envs.pause_at(idx)

                # indexing along the batch dimensions
                test_recurrent_hidden_states = test_recurrent_hidden_states[
                                               :, state_index
                                               ]
                not_done_masks = not_done_masks[state_index]
                current_episode_reward = current_episode_reward[
                    state_index
                ]

                for k, v in batch.items():
                    batch[k] = v[state_index]

                if args.video == 1:

                    rgb_frames = [rgb_frames[i] for i in state_index]

    episode_reward_mean = (episode_rewards / episode_counts).mean().item()
    episode_spl_mean = (episode_spls / episode_counts).mean().item()
    episode_success_mean = (episode_success / episode_counts).mean().item()

    print("Average episode reward: {:.6f}".format(episode_reward_mean))
    print("Average episode success: {:.6f}".format(episode_success_mean))
    print("Average episode spl: {:.6f}".format(episode_spl_mean))


if __name__ == "__main__":
    main()
