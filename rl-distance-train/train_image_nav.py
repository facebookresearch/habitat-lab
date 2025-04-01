import argparse
import random
import numpy as np
import torch
import os
import json

from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_baselines.config.default import get_config
from omegaconf import OmegaConf

# A function to build configuration for PPO training
def build_PPO_config(args):
    config = get_config("instance_imagenav/ddppo_instance_imagenav.yaml")
    OmegaConf.set_readonly(config, False)
    
    # Set checkpoint and tensorboard directories with time-stamped placeholders
    config.habitat_baselines.checkpoint_folder = (
        "rl-distance-train/PPO-Img-Nav-checkpoints/policy-${now:%Y-%m-%d}_${now:%H-%M-%S}"
    )
    config.habitat_baselines.tensorboard_dir = (
        "rl-distance-train/tb/PPO-Img-Nav-${now:%Y-%m-%d}_${now:%H-%M-%S}"
    )

    # Optionally, use depth-only configuration if specified
    if args.depth_only:
        config.habitat_baselines.rl.policy.main_agent.depth_only = True

    if args.dist_to_goal:
        del config.habitat.task.lab_sensors.gps_sensor
        del config.habitat.task.lab_sensors.compass_sensor
    else:
        del config.habitat.task.lab_sensors.pointgoal_with_gps_compass_sensor
    
    OmegaConf.set_readonly(config, True)
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth-only", action="store_true", default=False)
    parser.add_argument("--dist-to-goal", action="store_true", default=False)
    parser.add_argument("--eval", action="store_true", default=False)
    args = parser.parse_args()

    # Build the config for PPO
    config = build_PPO_config(args)

    # Create the checkpoint folder if it doesn't exist
    os.makedirs(config.habitat_baselines.checkpoint_folder, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.habitat_baselines.checkpoint_folder, "config.yaml"))

    # Set randomness
    random.seed(config.habitat.seed)
    np.random.seed(config.habitat.seed)
    torch.manual_seed(config.habitat.seed)
    if config.habitat_baselines.force_torch_single_threaded and torch.cuda.is_available():
        torch.set_num_threads(1)

    # Suppress logging for external libraries if needed
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"
    os.environ["MAGNUM_GPU_CONTEXT"] = "egl"

    # Build the trainer and run evaluation or training
    trainer = PPOTrainer(config)
    if args.eval:
        trainer.eval()
    else:
        trainer.train()
