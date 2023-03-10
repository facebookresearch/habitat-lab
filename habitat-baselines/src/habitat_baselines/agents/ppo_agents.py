#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import random
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from gym.spaces import Box
from gym.spaces import Dict as SpaceDict
from gym.spaces import Discrete
from omegaconf import DictConfig, OmegaConf

import habitat
from habitat.core.agent import Agent
from habitat.core.simulator import Observations
from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy
from habitat_baselines.utils.common import batch_obs


@dataclass
class PPOAgentConfig:
    INPUT_TYPE: str = "rgb"
    MODEL_PATH: str = "data/checkpoints/gibson-rgb-best.pth"
    RESOLUTION: int = 256
    HIDDEN_SIZE: int = 512
    RANDOM_SEED: int = 7
    PTH_GPU_ID: int = 0
    GOAL_SENSOR_UUID: str = "pointgoal_with_gps_compass"


def get_default_config() -> DictConfig:
    return OmegaConf.create(PPOAgentConfig())  # type: ignore[call-overload]


class PPOAgent(Agent):
    def __init__(self, config: DictConfig) -> None:
        spaces = {
            get_default_config().GOAL_SENSOR_UUID: Box(
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
        observation_spaces = SpaceDict(spaces)

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
            torch.backends.cudnn.deterministic = True  # type: ignore

        self.actor_critic = PointNavResNetPolicy(
            observation_space=observation_spaces,
            action_space=action_spaces,
            hidden_size=self.hidden_size,
            normalize_visual_inputs="rgb" in spaces,
        )
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
        self.test_recurrent_hidden_states = torch.zeros(
            1,
            self.actor_critic.net.num_recurrent_layers,
            self.hidden_size,
            device=self.device,
        )
        self.not_done_masks = torch.zeros(
            1, 1, device=self.device, dtype=torch.bool
        )
        self.prev_actions = torch.zeros(
            1, 1, dtype=torch.long, device=self.device
        )

    def act(self, observations: Observations) -> Dict[str, int]:
        batch = batch_obs([observations], device=self.device)
        with torch.no_grad():
            action_data = self.actor_critic.act(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=False,
            )
            self.test_recurrent_hidden_states = action_data.rnn_hidden_states
            #  Make masks not done till reset (end of episode) will be called
            self.not_done_masks.fill_(True)
            self.prev_actions.copy_(action_data.actions)  # type: ignore

        return {"action": action_data.env_actions[0][0].item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-type",
        default="rgb",
        choices=["blind", "rgb", "depth", "rgbd"],
    )
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument(
        "--task-config",
        type=str,
        default="habitat-lab/habitat/config/task/pointnav.yaml",
    )
    args = parser.parse_args()

    agent_config = get_default_config()
    agent_config.INPUT_TYPE = args.input_type
    if args.model_path is not None:
        agent_config.MODEL_PATH = args.model_path

    agent = PPOAgent(agent_config)
    benchmark = habitat.Benchmark(config_paths=args.task_config)
    metrics = benchmark.evaluate(agent)

    for k, v in metrics.items():
        habitat.logger.info("{}: {:.3f}".format(k, v))


if __name__ == "__main__":
    main()
