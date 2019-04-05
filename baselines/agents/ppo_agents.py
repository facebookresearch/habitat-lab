#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import random

import numpy as np
import torch
from gym.spaces import Discrete, Dict, Box

import habitat
from baselines.rl.ppo import Policy
from baselines.rl.ppo.utils import batch_obs
from habitat import Config
from habitat.core.agent import Agent


def get_defaut_config():
    c = Config()
    c.INPUT_TYPE = "blind"
    c.MODEL_PATH = "data/checkpoints/blind.pth"
    c.RESOLUTION = 256
    c.HIDDEN_SIZE = 512
    c.RANDOM_SEED = 7
    c.PTH_GPU_ID = 0
    return c


class PPOAgent(Agent):
    def __init__(self, config: Config):
        spaces = {
            "pointgoal": Box(
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

        self.device = torch.device("cuda:{}".format(config.PTH_GPU_ID))
        self.hidden_size = config.HIDDEN_SIZE

        random.seed(config.RANDOM_SEED)
        torch.random.manual_seed(config.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True

        self.actor_critic = Policy(
            observation_space=observation_spaces,
            action_space=action_spaces,
            hidden_size=self.hidden_size,
        )
        self.actor_critic.to(self.device)

        if config.MODEL_PATH:
            ckpt = torch.load(config.MODEL_PATH, map_location=self.device)
            #  Filter only actor_critic weights
            self.actor_critic.load_state_dict(
                {
                    k.replace("actor_critic.", ""): v
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

    def reset(self):
        self.test_recurrent_hidden_states = torch.zeros(
            1, self.hidden_size, device=self.device
        )
        self.not_done_masks = torch.zeros(1, 1, device=self.device)

    def act(self, observations):
        batch = batch_obs([observations])
        for sensor in batch:
            batch[sensor] = batch[sensor].to(self.device)

        with torch.no_grad():
            _, actions, _, self.test_recurrent_hidden_states = self.actor_critic.act(
                batch,
                self.test_recurrent_hidden_states,
                self.not_done_masks,
                deterministic=False,
            )
            #  Make masks not done till reset (end of episode) will be called
            self.not_done_masks = torch.ones(1, 1, device=self.device)

        return actions[0][0].item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-type",
        default="blind",
        choices=["blind", "rgb", "depth", "rgbd"],
    )
    parser.add_argument("--model-path", default="", type=str)
    parser.add_argument(
        "--task-config", type=str, default="tasks/pointnav.yaml"
    )
    args = parser.parse_args()

    config = get_defaut_config()
    config.INPUT_TYPE = args.input_type
    config.MODEL_PATH = args.model_path

    agent = PPOAgent(config)
    benchmark = habitat.Benchmark(args.task_config)
    metrics = benchmark.evaluate(agent)

    for k, v in metrics.items():
        habitat.logger.info("{}: {:.3f}".format(k, v))


if __name__ == "__main__":
    main()
