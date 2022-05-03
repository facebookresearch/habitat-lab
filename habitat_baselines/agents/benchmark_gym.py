#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import os
import os.path as osp
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

import gym.spaces as spaces
import numpy as np
import torch
from tqdm import tqdm

import habitat
from habitat.core.agent import Agent
from habitat.core.environments import get_env_class
from habitat.utils.env_utils import make_env_fn
from habitat.utils.gym_adapter import HabGymWrapper
from habitat_baselines.agents.mp_agents import (
    AgentComposition,
    IkMoveArm,
    SpaManipPick,
    SpaResetModule,
)
from habitat_baselines.agents.ppo_agents import PPOAgent
from habitat_baselines.config.default import get_config
from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy
from habitat_baselines.utils.common import (
    action_array_to_dict,
    batch_obs,
    generate_video,
    get_num_actions,
)


def compress_action(action):
    if "grip_action" in action["action_args"]:
        return np.concatenate(
            [
                action["action_args"]["arm_action"],
                action["action_args"]["grip_action"],
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
        self.observation_space = self._gym_env.observation_space
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
        all_rewards = []

        traj_obs = []
        traj_dones = []
        traj_next_obs = []
        traj_actions = []
        traj_episode_ids = []
        traj_rewards = []
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
                traj_actions.append(copy.deepcopy(action))
                traj_dones.append(False)
                traj_episode_ids.append(
                    int(self._env.current_episode.episode_id)
                )

                observations, reward, done, _ = self._gym_env.direct_hab_step(
                    action
                )
                traj_rewards.append(reward)

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
                all_rewards.extend(traj_rewards)
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
            traj_rewards = []

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
                all_obs_l = batch_obs(all_obs_l)  # type:ignore
                all_next_obs_l = batch_obs(all_next_obs_l)  # type:ignore
            compressed_actions = np.array(
                [compress_action(action) for action in all_actions]
            )
            torch.save(
                {
                    "done": torch.FloatTensor(all_dones),
                    "obs": all_obs_l,
                    "next_obs": all_next_obs_l,
                    "rewards": torch.FloatTensor(all_rewards),
                    "episode_ids": torch.tensor(all_episode_ids),
                    "actions": torch.tensor(compressed_actions),
                },
                self._traj_save_path,
            )
            print(f"Saved trajectories to {self._traj_save_path}")

        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}
        pbar.close()

        return avg_metrics


class RearrangePPOAgent(PPOAgent):
    def __init__(self, config, obs_space, action_space) -> None:
        self.device = (
            torch.device("cuda:{}".format(config.PTH_GPU_ID))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.hidden_size = config.RL.PPO.hidden_size

        self.actor_critic = PointNavResNetPolicy.from_config(
            config, obs_space, action_space
        )
        self.action_space = action_space
        self.actor_critic.to(self.device)
        if config.MODEL_PATH:
            ckpt = torch.load(config.MODEL_PATH, map_location=self.device)
            #  Filter only actor_critic weights
            self.actor_critic.load_state_dict(
                {  # type: ignore
                    k[len("actor_critic.") :]: v
                    for k, v in ckpt["state_dict"].items()
                    if "actor_critic" in k
                }
            )

        else:
            habitat.logger.error(
                "Model checkpoint wasn't loaded, evaluating " "a random model."
            )

        self.test_recurrent_hidden_states: Optional[torch.Tensor] = None
        self.not_done_masks: Optional[torch.Tensor] = None
        self.prev_actions: Optional[torch.Tensor] = None

    def reset(self) -> None:
        super().reset()
        self.prev_actions = torch.zeros(
            1,
            get_num_actions(self.action_space),
            dtype=torch.float32,
            device=self.device,
        )

    def act(self, observations):
        batch = batch_obs([observations], device=self.device)
        with torch.no_grad():
            (
                _,
                actions,
                _,
                self.test_recurrent_hidden_states,
            ) = self.actor_critic.act(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=False,
            )
            #  Make masks not done till reset (end of episode) will be called
            self.not_done_masks.fill_(True)
            self.prev_actions.copy_(actions)  # type: ignore
        action = action_array_to_dict(self.action_space, actions[0])

        return action["action"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill-type", default="mp_reach")
    parser.add_argument("--num-eval", type=int, default=None)
    parser.add_argument("--traj-save-path", type=str, default=None)
    parser.add_argument("--model-load-path", type=str, default=None)
    parser.add_argument(
        "--task-cfg",
        type=str,
        default="habitat_baselines/config/rearrange/spap_rearrangepick.yaml",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()

    config = get_config(args.task_cfg, args.opts)

    def should_save(metrics):
        was_success = metrics[config.RL.SUCCESS_MEASURE]
        return was_success

    benchmark = BenchmarkGym(
        config,
        config.VIDEO_OPTION,
        config.VIDEO_DIR,
        {config.RL.SUCCESS_MEASURE},
        args.traj_save_path,
        should_save_fn=should_save,
    )

    ac_cfg = config.TASK_CONFIG.TASK.ACTIONS
    env = benchmark._env

    use_skill: Agent = None

    if args.skill_type == "mp_reach":

        def get_arm_rest_args(skill):
            return {"robot_target": skill._task.desired_resting}

        use_skill = IkMoveArm(
            env,
            config.SENSE_PLAN_ACT,
            ac_cfg,
            auto_get_args_fn=get_arm_rest_args,
        )
    elif args.skill_type == "mp_pick":

        def get_object_args(skill):
            target_idx = skill._sim.get_targets()[0][0]
            return {"obj": target_idx}

        use_skill = AgentComposition(
            [
                SpaManipPick(
                    env,
                    config.SENSE_PLAN_ACT,
                    ac_cfg,
                    auto_get_args_fn=get_object_args,
                ),
                SpaResetModule(
                    env,
                    config.SENSE_PLAN_ACT,
                    ac_cfg,
                    ignore_first=True,
                    auto_get_args_fn=get_object_args,
                ),
            ],
            env,
            config.SENSE_PLAN_ACT,
            ac_cfg,
            auto_get_args_fn=get_object_args,
        )
    elif args.skill_type == "nn_policy":
        config.defrost()
        config.MODEL_PATH = args.model_load_path
        config.PTH_GPU_ID = 0
        config.freeze()

        use_skill = RearrangePPOAgent(
            config, benchmark.observation_space, env.action_space
        )
    else:
        raise ValueError(f"Unrecognized skill {args.skill_type}")

    metrics = benchmark.evaluate(use_skill, args.num_eval)
    for k, v in metrics.items():
        habitat.logger.info("{}: {:.3f}".format(k, v))


if __name__ == "__main__":
    main()
