#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import time

import cv2
import numpy as np
import torch
from torch.utils import tensorboard

import habitat
from config.default import get_config as cfg_baseline
from habitat import logger
from habitat.config.default import get_config
from habitat.utils.visualizations.utils import images_to_video
from rl.ppo import PPO, Policy
from rl.ppo.utils import batch_obs, frames_to_tb_video, generate_frame
from train_ppo import make_env_fn


def poll_checkpoint_folder(checkpoint_folder, previous_ckpt_ind):
    assert os.path.isdir(checkpoint_folder), "invalid checkpoint folder path"
    models = os.listdir(checkpoint_folder)
    models.sort(key=lambda x: int(x.strip().split(".")[1]))
    ind = previous_ckpt_ind + 1
    if ind < len(models):
        return os.path.join(checkpoint_folder, models[ind])
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=False)
    parser.add_argument("--tracking-model-dir", type=str, required=False)
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
    parser.add_argument("--video", type=int, default=0, choices=[0, 1])
    parser.add_argument("--out-dir-video", type=str)
    parser.add_argument("--tensorboard-dir", type=str, default="tb_eval")

    args = parser.parse_args()
    assert (args.model_path is not None) != (
        args.tracking_model_dir is not None
    ), "Must specify a single model or a directory of models, but not both"

    writer_kwargs = dict(
        log_dir=args.tensorboard_dir, purge_step=0, flush_secs=30
    )
    with (tensorboard.SummaryWriter(**writer_kwargs)) as writer:

        if args.model_path is not None:
            eval_checkpoint(args.model_path, args, writer)
        else:  # track model progression
            prev_ckpt_ind = -1
            while True:
                current_ckpt = None
                while current_ckpt is None:
                    current_ckpt = poll_checkpoint_folder(
                        args.tracking_model_dir, prev_ckpt_ind
                    )
                    time.sleep(2)  # sleep for 2 seconds before polling again

                logger.warning(
                    "=============current_ckpt: {}=============".format(
                        current_ckpt
                    )
                )
                eval_checkpoint(current_ckpt, args, writer)
                prev_ckpt_ind += 1


def eval_checkpoint(checkpoint_path, args, writer):
    checkpoint_idx = int(checkpoint_path.strip().split(".")[1])
    env_configs = []
    baseline_configs = []
    device = torch.device("cuda:{}".format(args.pth_gpu_id))

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

    ckpt = torch.load(checkpoint_path, map_location=device)

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

    rgb_frames = None
    if args.video == 1:
        rgb_frames = [[]] * args.num_processes
        os.makedirs(args.out_dir_video, exist_ok=True)

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
                # episode ended, record and generate videos
                stats_episodes.add(current_episodes[i].episode_id)
                if args.video == 1 and len(rgb_frames[i]) > 0:
                    video_name = "episode{}_ckpt{}_spl{:.2f}".format(
                        current_episodes[i].episode_id,
                        checkpoint_idx,
                        infos[i]["spl"],
                    )
                    images_to_video(
                        rgb_frames[i], args.out_dir_video, video_name
                    )
                    frames_to_tb_video(
                        "episode{}".format(current_episodes[i].episode_id),
                        checkpoint_idx,
                        rgb_frames[i],
                        writer,
                        fps=10,
                    )
                    rgb_frames[i] = []

            elif args.video == 1:
                # episode continues, record current frame
                frame = generate_frame(observations[i], infos[i])
                rgb_frames[i].append(frame)

        if len(envs_to_pause) > 0:
            # stop tracking ended episodes

            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            # indexing along the batch dimensions
            test_recurrent_hidden_states = test_recurrent_hidden_states[
                :, state_index
            ]
            not_done_masks = not_done_masks[state_index]
            current_episode_reward = current_episode_reward[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

            if args.video == 1:
                rgb_frames = [rgb_frames[i] for i in state_index]

    episode_reward_mean = (episode_rewards / episode_counts).mean().item()
    episode_spl_mean = (episode_spls / episode_counts).mean().item()
    episode_success_mean = (episode_success / episode_counts).mean().item()

    logger.info("Average episode reward: {:.6f}".format(episode_reward_mean))
    logger.info("Average episode success: {:.6f}".format(episode_success_mean))
    logger.info("Average episode SPL: {:.6f}".format(episode_spl_mean))

    writer.add_scalar("Average reward", episode_reward_mean, checkpoint_idx)
    writer.add_scalar("SPL", episode_spl_mean, checkpoint_idx)


if __name__ == "__main__":
    main()
