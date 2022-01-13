#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

import gym.spaces as spaces
import numpy as np
import torch
from tqdm import tqdm

from habitat.core.agent import Agent
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.utils.common import batch_obs, generate_video
from habitat_baselines.utils.env_utils import make_env_fn
from habitat_baselines.utils.gym_adapter import HabGymWrapper


def compress_action(action):
    if "grip_action" in action["action_args"]:
        return np.concatenate(
            [
                action["action_args"]["arm_action"],
                np.array(
                    [action["action_args"]["grip_action"]],
                    dtype=np.float32,
                ),
            ]
        )
    else:
        return action["action_args"]["arm_action"]


class BenchmarkGym:
    def __init__(
        self,
        config: Any,
        video_option: List[str],
        video_dir: str,
        vid_filename_metrics: Set[str],
        traj_save_dir: str = None,
        should_save_fn=None,
        writer=None,
    ) -> None:

        env_class = get_env_class(config.ENV_NAME)

        env = make_env_fn(env_class=env_class, config=config)
        self._gym_env = HabGymWrapper(env, save_orig_obs=True)
        self._video_option = video_option
        self._video_dir = video_dir
        self._writer = writer
        self._vid_filename_metrics = vid_filename_metrics
        self._traj_save_path = traj_save_dir
        self._should_save_fn = should_save_fn

    @property
    def _env(self):
        return self._gym_env._env._env

    def evaluate(
        self,
        agent: "Agent",
        num_episodes: Optional[int] = None,
    ) -> Dict[str, float]:
        if num_episodes is None:
            num_episodes = len(self._env.episodes)
        else:
            assert num_episodes <= len(self._env.episodes), (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(
                    num_episodes, len(self._env.episodes)
                )
            )

        assert num_episodes > 0, "num_episodes should be greater than 0"

        agg_metrics: Dict = defaultdict(float)
        rgb_frames = []
        should_render = len(self._video_option) > 0

        count_episodes = 0
        all_dones = []
        all_obs_l = []
        all_next_obs_l = []
        all_actions = []
        all_episode_ids = []

        traj_obs = []
        traj_dones = []
        traj_next_obs = []
        traj_actions = []
        traj_episode_ids = []
        pbar = tqdm(total=num_episodes)

        while count_episodes < num_episodes:
            observations = self._gym_env.reset()
            agent.reset()
            if should_render:
                rgb_frames.append(self._gym_env.render())

            done = False

            while not done:
                traj_obs.append(observations)

                action = agent.act(self._gym_env.orig_obs)
                traj_actions.append(action)
                traj_dones.append(False)
                traj_episode_ids.append(
                    int(self._env.current_episode.episode_id)
                )

                observations, _, done, _ = self._gym_env.direct_hab_step(
                    action
                )

                traj_next_obs.append(observations)

                if should_render:
                    rgb_frames.append(self._gym_env.render())

            traj_dones[-1] = True

            metrics = self._env.get_metrics()
            metrics["length"] = len(traj_obs)
            if self._should_save_fn is None or self._should_save_fn(metrics):
                assert sum(traj_dones) == 1
                all_obs_l.extend(traj_obs)
                all_dones.extend(traj_dones)
                all_next_obs_l.extend(traj_next_obs)
                all_actions.extend(traj_actions)
                all_episode_ids.extend(traj_episode_ids)

                count_episodes += 1
                pbar.update(1)

            traj_obs = []
            traj_dones = []
            traj_next_obs = []
            traj_actions = []
            traj_episode_ids = []

            metrics = self._env.get_metrics()
            for m, v in metrics.items():
                if isinstance(v, dict):
                    for sub_m, sub_v in v.items():
                        agg_metrics[m + "/" + str(sub_m)] += sub_v
                else:
                    agg_metrics[m] += v

            if should_render:
                generate_video(
                    video_option=self._video_option,
                    video_dir=self._video_dir,
                    images=rgb_frames,
                    episode_id=self._env.current_episode.episode_id,
                    checkpoint_idx=0,
                    metrics={
                        k: v
                        for k, v in metrics.items()
                        if k in self._vid_filename_metrics
                    },
                    tb_writer=self._writer,
                    verbose=False,
                )

        if self._traj_save_path is not None:
            save_dir = osp.dirname(self._traj_save_path)
            os.makedirs(save_dir, exist_ok=True)
            if isinstance(self._gym_env.observation_space, spaces.Dict):
                all_obs = batch_obs(all_obs_l)  # type:ignore
                all_next_obs = batch_obs(all_next_obs_l)  # type:ignore
            torch.save(
                {
                    "done": torch.FloatTensor(all_dones),
                    "obs": all_obs,
                    "next_obs": all_next_obs,
                    "episode_ids": all_episode_ids,
                    "actions": torch.tensor(
                        [compress_action(action) for action in all_actions]
                    ),
                },
                self._traj_save_path,
            )
            print(f"Saved trajectories to {self._traj_save_path}")

        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}
        pbar.close()

        return avg_metrics
