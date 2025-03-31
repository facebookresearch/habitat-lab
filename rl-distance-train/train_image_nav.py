import argparse
import random
import numpy as np
import torch
import os

from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_baselines.config.default import get_config
from omegaconf import OmegaConf

# A function to build configuration for PPO training
def build_PPO_config(args):
    config = get_config("instance_imagenav/ddppo_instance_imagenav.yaml")

    OmegaConf.set_readonly(config, False)
    config.habitat_baselines.checkpoint_folder = "rl-distance-train/PPO-Img-Nav-checkpoints/policy-${now:%Y-%m-%d}_${now:%H-%M-%S}"
    config.habitat_baselines.tensorboard_dir = "rl-distance-train/tb/PPO-Img-Nav-${now:%Y-%m-%d}_${now:%H-%M-%S}"
    if args.depth_only or args.use_pretrained_encoder:
        config.habitat_baselines.rl.policy.main_agent.depth_only = True

    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth-only", action="store_true", default=False)
    args = parser.parse_args()
    config = build_PPO_config(args)  # Build the config for PPO
    # Set randomness
    random.seed(config.habitat.seed)
    np.random.seed(config.habitat.seed)
    torch.manual_seed(config.habitat.seed)
    if (
        config.habitat_baselines.force_torch_single_threaded
        and torch.cuda.is_available()
    ):
        torch.set_num_threads(1)

    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"
    os.environ["MAGNUM_GPU_CONTEXT"] = "egl"

    # Build the trainer and start training
    trainer = PPOTrainer(config)
    trainer.train()
