#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import random

import numpy as np
import torch
from gym.spaces import Box, Dict, Discrete

import habitat
from habitat.config import Config
from habitat.config.default import get_config
from habitat.core.agent import Agent
from habitat_baselines.common.utils import batch_obs
from habitat_baselines.rl.ppo import PointNavBaselinePolicy


def get_default_config():
    c = Config()
    c.INPUT_TYPE = "blind"
    c.MODEL_PATH = "data/checkpoints/blind.pth"
    c.RESOLUTION = 256
    c.HIDDEN_SIZE = 512
    c.RANDOM_SEED = 7
    c.PTH_GPU_ID = 0
    c.GOAL_SENSOR_UUID = "pointgoal_with_gps_compass"
    return c


class PPOAgent(Agent):
    def __init__(self, config: Config):
        self.goal_sensor_uuid = config.GOAL_SENSOR_UUID
        spaces = {
            self.goal_sensor_uuid: Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,),
                dtype=np.float32,
            )
        }

        if config.INPUT_TYPE in ["depth", "rgbd"]:
            spaces["depth"] = Box(
                low=0,
                high=1,
                shape=(config.RESOLUTION, config.RESOLUTION, 1),
                dtype=np.float32,
            )

        if config.INPUT_TYPE in ["rgb", "rgbd"]:
            spaces["rgb"] = Box(
                low=0,
                high=255,
                shape=(config.RESOLUTION, config.RESOLUTION, 3),
                dtype=np.uint8,
            )
        observation_spaces = Dict(spaces)

        action_spaces = Discrete(4)

        self.device = (
            torch.device("cuda:{}".format(config.PTH_GPU_ID))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.hidden_size = config.HIDDEN_SIZE

        random.seed(config.RANDOM_SEED)
        torch.random.manual_seed(config.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True

        self.actor_critic = PointNavBaselinePolicy(
            observation_space=observation_spaces,
            action_space=action_spaces,
            hidden_size=self.hidden_size,
            goal_sensor_uuid=self.goal_sensor_uuid,
        )
        self.actor_critic.to(self.device)

        if config.MODEL_PATH:
            ckpt = torch.load(config.MODEL_PATH, map_location=self.device)
            #  Filter only actor_critic weights
            self.actor_critic.load_state_dict(
                {
                    k[len("actor_critic.") :]: v
                    for k, v in ckpt["state_dict"].items()
                    if "actor_critic" in k
                }
            )

        else:
            habitat.logger.error(
                "Model checkpoint wasn't loaded, evaluating " "a random model."
            )

        self.test_recurrent_hidden_states = None
        self.not_done_masks = None
        self.prev_actions = None

    def reset(self):
        self.test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            1,
            self.hidden_size,
            device=self.device,
        )
        self.not_done_masks = torch.zeros(1, 1, device=self.device)
        self.prev_actions = torch.zeros(
            1, 1, dtype=torch.long, device=self.device
        )

    def act(self, observations):
        batch = batch_obs([observations])
        for sensor in batch:
            batch[sensor] = batch[sensor].to(self.device)

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
            self.not_done_masks = torch.ones(1, 1, device=self.device)
            self.prev_actions.copy_(actions)

        return {"action": actions[0][0].item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-type",
        default="blind",
        choices=["blind", "rgb", "depth", "rgbd"],
    )
    parser.add_argument("--model-path", default="", type=str)
    parser.add_argument(
        "--task-config", type=str, default="configs/tasks/pointnav.yaml"
    )
    args = parser.parse_args()

    config = get_config(args.task_config)

    agent_config = get_default_config()
    agent_config.INPUT_TYPE = args.input_type
    agent_config.MODEL_PATH = args.model_path
    agent_config.GOAL_SENSOR_UUID = config.TASK.GOAL_SENSOR_UUID

    agent = PPOAgent(agent_config)
    benchmark = habitat.Benchmark(config_paths=args.task_config)
    metrics = benchmark.evaluate(agent)

    for k, v in metrics.items():
        habitat.logger.info("{}: {:.3f}".format(k, v))


if __name__ == "__main__":
    main()
