#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections import deque
from typing import Dict, List

import numpy as np
import torch

from habitat import Config, logger
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import NavRLEnv
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensorboard_utils import (
    TensorboardWriter,
    get_tensorboard_writer,
)
from habitat_baselines.common.utils import (
    batch_obs,
    generate_video,
    poll_checkpoint_folder,
    update_linear_schedule,
)
from habitat_baselines.rl.ppo import PPO, Policy


@baseline_registry.register_trainer(name="ppo")
class PPOTrainer(BaseRLTrainer):
    r"""
    Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347
    """
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        self.device = None
        self.video_option = []
        if config is not None:
            logger.info(f"config: {config}")

    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""
        Sets up actor critic and agent for PPO
        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(ppo_cfg.log_file)

        self.actor_critic = Policy(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            hidden_size=512,
            goal_sensor_uuid=self.config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
        )
        self.actor_critic.to(self.device)

        self.agent = PPO(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
        )

    def save_checkpoint(self, file_name: str) -> None:
        r"""
        Save checkpoint with specified name
        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        torch.save(
            checkpoint,
            os.path.join(
                self.config.TRAINER.RL.PPO.checkpoint_folder, file_name
            ),
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""
        Load checkpoint of specified path as a dict
        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, map_location=self.device)

    def train(self) -> None:
        r"""
        Main method for training PPO
        Returns:
            None
        """
        assert (
            self.config is not None
        ), "trainer is not properly initialized, need to specify config file"

        self.envs = construct_envs(self.config, NavRLEnv)

        ppo_cfg = self.config.TRAINER.RL.PPO
        self.device = torch.device("cuda", ppo_cfg.pth_gpu_id)
        if not os.path.isdir(ppo_cfg.checkpoint_folder):
            os.makedirs(ppo_cfg.checkpoint_folder)
        self._setup_actor_critic_agent(ppo_cfg)
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        observations = self.envs.reset()
        batch = batch_obs(observations)

        rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            self.envs.action_spaces[0],
            ppo_cfg.hidden_size,
        )
        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])
        rollouts.to(self.device)

        episode_rewards = torch.zeros(self.envs.num_envs, 1)
        episode_counts = torch.zeros(self.envs.num_envs, 1)
        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        window_episode_reward = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_counts = deque(maxlen=ppo_cfg.reward_window_size)

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0

        with (
            get_tensorboard_writer(
                log_dir=ppo_cfg.tensorboard_dir,
                purge_step=count_steps,
                flush_secs=30,
            )
        ) as writer:
            for update in range(ppo_cfg.num_updates):
                if ppo_cfg.use_linear_lr_decay:
                    update_linear_schedule(
                        self.agent.optimizer,
                        update,
                        ppo_cfg.num_updates,
                        ppo_cfg.lr,
                    )

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * (
                        1 - update / ppo_cfg.num_updates
                    )

                for step in range(ppo_cfg.num_steps):
                    t_sample_action = time.time()
                    # sample actions
                    with torch.no_grad():
                        step_observation = {
                            k: v[step]
                            for k, v in rollouts.observations.items()
                        }

                        (
                            values,
                            actions,
                            actions_log_probs,
                            recurrent_hidden_states,
                        ) = self.actor_critic.act(
                            step_observation,
                            rollouts.recurrent_hidden_states[step],
                            rollouts.masks[step],
                        )
                    pth_time += time.time() - t_sample_action

                    t_step_env = time.time()

                    outputs = self.envs.step([a[0].item() for a in actions])
                    observations, rewards, dones, infos = [
                        list(x) for x in zip(*outputs)
                    ]

                    env_time += time.time() - t_step_env

                    t_update_stats = time.time()
                    batch = batch_obs(observations)
                    rewards = torch.tensor(rewards, dtype=torch.float)
                    rewards = rewards.unsqueeze(1)

                    masks = torch.tensor(
                        [[0.0] if done else [1.0] for done in dones],
                        dtype=torch.float,
                    )

                    current_episode_reward += rewards
                    episode_rewards += (1 - masks) * current_episode_reward
                    episode_counts += 1 - masks
                    current_episode_reward *= masks

                    rollouts.insert(
                        batch,
                        recurrent_hidden_states,
                        actions,
                        actions_log_probs,
                        values,
                        rewards,
                        masks,
                    )

                    count_steps += self.envs.num_envs
                    pth_time += time.time() - t_update_stats

                window_episode_reward.append(episode_rewards.clone())
                window_episode_counts.append(episode_counts.clone())

                t_update_model = time.time()
                with torch.no_grad():
                    last_observation = {
                        k: v[-1] for k, v in rollouts.observations.items()
                    }
                    next_value = self.actor_critic.get_value(
                        last_observation,
                        rollouts.recurrent_hidden_states[-1],
                        rollouts.masks[-1],
                    ).detach()

                rollouts.compute_returns(
                    next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
                )

                value_loss, action_loss, dist_entropy = self.agent.update(
                    rollouts
                )

                rollouts.after_update()
                pth_time += time.time() - t_update_model

                losses = [value_loss, action_loss]
                stats = zip(
                    ["count", "reward"],
                    [window_episode_counts, window_episode_reward],
                )
                deltas = {
                    k: (
                        (v[-1] - v[0]).sum().item()
                        if len(v) > 1
                        else v[0].sum().item()
                    )
                    for k, v in stats
                }
                deltas["count"] = max(deltas["count"], 1.0)

                writer.add_scalar(
                    "reward", deltas["reward"] / deltas["count"], count_steps
                )

                writer.add_scalars(
                    "losses",
                    {k: l for l, k in zip(losses, ["value", "policy"])},
                    count_steps,
                )

                # log stats
                if update > 0 and update % ppo_cfg.log_interval == 0:
                    logger.info(
                        "update: {}\tfps: {:.3f}\t".format(
                            update, count_steps / (time.time() - t_start)
                        )
                    )

                    logger.info(
                        "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                        "frames: {}".format(
                            update, env_time, pth_time, count_steps
                        )
                    )

                    window_rewards = (
                        window_episode_reward[-1] - window_episode_reward[0]
                    ).sum()
                    window_counts = (
                        window_episode_counts[-1] - window_episode_counts[0]
                    ).sum()

                    if window_counts > 0:
                        logger.info(
                            "Average window size {} reward: {:3f}".format(
                                len(window_episode_reward),
                                (window_rewards / window_counts).item(),
                            )
                        )
                    else:
                        logger.info("No episodes finish in current window")

                # checkpoint model
                if update % ppo_cfg.checkpoint_interval == 0:
                    self.save_checkpoint(f"ckpt.{count_checkpoints}.pth")
                    count_checkpoints += 1

    def eval(self) -> None:
        r"""
        Main method of evaluating PPO
        Returns:
            None
        """
        ppo_cfg = self.config.TRAINER.RL.PPO
        self.device = torch.device("cuda", ppo_cfg.pth_gpu_id)
        self.video_option = ppo_cfg.video_option.strip().split(",")

        if "tensorboard" in self.video_option:
            assert (
                ppo_cfg.tensorboard_dir is not None
            ), "Must specify a tensorboard directory for video display"
        if "disk" in self.video_option:
            assert (
                ppo_cfg.video_dir is not None
            ), "Must specify a directory for storing videos on disk"

        with get_tensorboard_writer(
            ppo_cfg.tensorboard_dir, purge_step=0, flush_secs=30
        ) as writer:
            if os.path.isfile(ppo_cfg.eval_ckpt_path_or_dir):
                # evaluate singe checkpoint
                self._eval_checkpoint(ppo_cfg.eval_ckpt_path_or_dir, writer)
            else:
                # evaluate multiple checkpoints in order
                prev_ckpt_ind = -1
                while True:
                    current_ckpt = None
                    while current_ckpt is None:
                        current_ckpt = poll_checkpoint_folder(
                            ppo_cfg.eval_ckpt_path_or_dir, prev_ckpt_ind
                        )
                        time.sleep(2)  # sleep for 2 secs before polling again
                    logger.warning(
                        "=============current_ckpt: {}=============".format(
                            current_ckpt
                        )
                    )
                    prev_ckpt_ind += 1
                    self._eval_checkpoint(
                        checkpoint_path=current_ckpt,
                        writer=writer,
                        cur_ckpt_idx=prev_ckpt_ind,
                    )

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        cur_ckpt_idx: int = 0,
    ) -> None:
        r"""
        Evaluates a single checkpoint
        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            cur_ckpt_idx: index of cur checkpoint for logging

        Returns:
            None
        """
        ckpt_dict = self.load_checkpoint(
            checkpoint_path, map_location=self.device
        )

        ckpt_config = ckpt_dict["config"]
        config = self.config.clone()
        ckpt_cmd_opts = ckpt_config.CMD_TRAILING_OPTS
        eval_cmd_opts = config.CMD_TRAILING_OPTS

        # config merge priority: eval_opts > ckpt_opts > eval_cfg > ckpt_cfg
        # first line for old checkpoint compatibility
        config.merge_from_other_cfg(ckpt_config)
        config.merge_from_other_cfg(self.config)
        config.merge_from_list(ckpt_cmd_opts)
        config.merge_from_list(eval_cmd_opts)

        ppo_cfg = config.TRAINER.RL.PPO
        config.TASK_CONFIG.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = "val"
        agent_sensors = ppo_cfg.sensors.strip().split(",")
        config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = agent_sensors
        if self.video_option:
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
        config.freeze()

        logger.info(f"env config: {config}")
        self.envs = construct_envs(config, NavRLEnv)
        self._setup_actor_critic_agent(ppo_cfg)

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic

        observations = self.envs.reset()
        batch = batch_obs(observations)
        for sensor in batch:
            batch[sensor] = batch[sensor].to(self.device)

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        test_recurrent_hidden_states = torch.zeros(
            ppo_cfg.num_processes, ppo_cfg.hidden_size, device=self.device
        )
        not_done_masks = torch.zeros(
            ppo_cfg.num_processes, 1, device=self.device
        )
        stats_episodes = dict()  # dict of dicts that stores stats per episode

        rgb_frames = [
            []
        ] * ppo_cfg.num_processes  # type: List[List[np.ndarray]]
        if self.video_option:
            os.makedirs(ppo_cfg.video_dir, exist_ok=True)

        while (
            len(stats_episodes) < ppo_cfg.count_test_episodes
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                _, actions, _, test_recurrent_hidden_states = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    not_done_masks,
                    deterministic=False,
                )

            outputs = self.envs.step([a[0].item() for a in actions])

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(observations)
            for sensor in batch:
                batch[sensor] = batch[sensor].to(self.device)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            rewards = torch.tensor(
                rewards, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not_done_masks[i].item() == 0:
                    episode_stats = dict()
                    episode_stats["spl"] = infos[i]["spl"]
                    episode_stats["success"] = int(infos[i]["spl"] > 0)
                    episode_stats["reward"] = current_episode_reward[i].item()
                    current_episode_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats
                    if self.video_option:
                        generate_video(
                            ppo_cfg,
                            rgb_frames[i],
                            current_episodes[i].episode_id,
                            cur_ckpt_idx,
                            infos[i]["spl"],
                            writer,
                        )
                        rgb_frames[i] = []

                # episode continues
                elif self.video_option:
                    frame = observations_to_image(observations[i], infos[i])
                    rgb_frames[i].append(frame)

            # pausing self.envs with no new episode
            if len(envs_to_pause) > 0:
                state_index = list(range(self.envs.num_envs))
                for idx in reversed(envs_to_pause):
                    state_index.pop(idx)
                    self.envs.pause_at(idx)

                # indexing along the batch dimensions
                test_recurrent_hidden_states = test_recurrent_hidden_states[
                    state_index
                ]
                not_done_masks = not_done_masks[state_index]
                current_episode_reward = current_episode_reward[state_index]

                for k, v in batch.items():
                    batch[k] = v[state_index]

                if self.video_option:
                    rgb_frames = [rgb_frames[i] for i in state_index]

        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = sum(
                [v[stat_key] for v in stats_episodes.values()]
            )
        num_episodes = len(stats_episodes)

        episode_reward_mean = aggregated_stats["reward"] / num_episodes
        episode_spl_mean = aggregated_stats["spl"] / num_episodes
        episode_success_mean = aggregated_stats["success"] / num_episodes

        logger.info(
            "Average episode reward: {:.6f}".format(episode_reward_mean)
        )
        logger.info(
            "Average episode success: {:.6f}".format(episode_success_mean)
        )
        logger.info("Average episode SPL: {:.6f}".format(episode_spl_mean))

        writer.add_scalars(
            "eval_reward",
            {"average reward": episode_reward_mean},
            cur_ckpt_idx,
        )
        writer.add_scalars(
            "eval_SPL", {"average SPL": episode_spl_mean}, cur_ckpt_idx
        )
        writer.add_scalars(
            "eval_success",
            {"average success": episode_success_mean},
            cur_ckpt_idx,
        )
