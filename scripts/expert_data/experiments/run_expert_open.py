import numpy as np
import torch
from omegaconf import OmegaConf

from scripts.expert_data.envs.murp_env import MurpEnv
from scripts.expert_data.policies.heuristic_open_policy import (
    HeuristicOpenPolicy,
)


def main(config):
    config = OmegaConf.load(config)
    murp_env = MurpEnv(config)
    policy = HeuristicOpenPolicy(murp_env)

    hand = config.hand

    murp_env.reset_robot(config.target_name)

    grip_iters, open_iters, move_iters, _ = policy.TARGET_CONFIG[
        config.target_name
    ]
    if config.hand == "left":
        move_iters = None
    policy.execute_grasp_sequence(
        config.hand, grip_iters, open_iters, move_iters
    )
    print("saved video to: ", murp_env.save_path)


if __name__ == "__main__":
    config = "scripts/expert_data/configs/config.yaml"
    main(config)
