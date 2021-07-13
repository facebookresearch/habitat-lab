import os
import os.path as osp
from collections import defaultdict
from typing import Dict, List, Optional, Set

import numpy as np
import torch

from habitat.core.agent import Agent
from habitat.core.benchmark import Benchmark
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.utils.common import batch_obs, generate_video


class BenchmarkRenderer(Benchmark):
    def __init__(
        self,
        config_paths: Optional[str],
        video_option: List[str],
        video_dir: str,
        vid_filename_metrics: Set[str],
        traj_save_dir: str = None,
        writer=None,
    ) -> None:
        super().__init__(config_paths, False)
        self._video_option = video_option
        self._video_dir = video_dir
        self._writer = writer
        self._vid_filename_metrics = vid_filename_metrics
        self._traj_save_path = traj_save_dir

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
        dones = []
        obs = []
        next_obs = []
        actions = []
        episode_ids = []

        while count_episodes < num_episodes:
            observations = self._env.reset()
            agent.reset()
            if should_render:
                frame = observations_to_image(
                    observations, self._env.get_metrics()
                )
                rgb_frames.append(frame)

            while not self._env.episode_over:
                obs.append(observations)

                action = agent.act(observations)
                actions.append(action)
                dones.append(False)
                episode_ids.append(int(self._env.current_episode.episode_id))

                observations = self._env.step(action)

                next_obs.append(observations)

                if should_render:
                    frame = observations_to_image(
                        observations, self._env.get_metrics()
                    )
                    rgb_frames.append(frame)
            dones[-1] = True

            metrics = self._env.get_metrics()
            for m, v in metrics.items():
                if isinstance(v, dict):
                    for sub_m, sub_v in v.items():
                        agg_metrics[m + "/" + str(sub_m)] += sub_v
                else:
                    agg_metrics[m] += v

            def compress_action(action):
                return np.concatenate(
                    [
                        action["action_args"]["arm_action"],
                        np.array(
                            [action["action_args"]["grip_action"]],
                            dtype=np.float32,
                        ),
                    ]
                )

            if should_render:
                generate_video(
                    video_option=self._video_option,
                    video_dir=self._video_dir,
                    images=rgb_frames,
                    episode_id=None,
                    # episode_id=self._env.episode_id,
                    checkpoint_idx=0,
                    metrics={
                        k: v
                        for k, v in metrics.items()
                        if k in self._vid_filename_metrics
                    },
                    tb_writer=self._writer,
                )

            count_episodes += 1

        if self._traj_save_path is not None:
            save_dir = osp.dirname(self._traj_save_path)
            if not osp.exists(save_dir):
                os.makedirs(save_dir)
            obs = batch_obs(obs)
            next_obs = batch_obs(next_obs)
            torch.save(
                {
                    "done": torch.FloatTensor(dones),
                    "obs": obs,
                    "next_obs": next_obs,
                    "episode_ids": episode_ids,
                    "actions": torch.tensor(
                        [compress_action(action) for action in actions]
                    ),
                },
                self._traj_save_path,
            )

        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

        return avg_metrics
