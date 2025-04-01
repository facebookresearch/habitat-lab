import argparse
import random
import numpy as np
import torch
import os

from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_baselines.config.default import get_config
from omegaconf import OmegaConf

def build_PPO_config(args):
    config = get_config("instance_imagenav/ddppo_instance_imagenav_aux.yaml")

    OmegaConf.set_readonly(config, False)
    config.habitat_baselines.checkpoint_folder = (
        f"rl-distance-train/PPO-Dist-Img-Nav-checkpoints/policy-noise-{args.noise_coeff}"
    )
    config.habitat_baselines.tensorboard_dir = (
        f"rl-distance-train/tb/PPO-Dist-Img-Nav-{args.noise_coeff}"
    )
    config.habitat_baselines.rl.policy.main_agent.name = "PointNavGoalDistancePolicy"
    config.habitat_baselines.rl.policy.main_agent.noise_coefficient = args.noise_coeff
    config.habitat_baselines.rl.ddppo.rnn_type = "GRU"
    del config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
    
    OmegaConf.set_readonly(config, True)
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_coeff", type=float, default=0.1)
    parser.add_argument("--eval", action="store_true", default=False)
    args = parser.parse_args()

    # Set up distributed training device allocation using the LOCAL_RANK env variable.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    # Optionally, get global rank if needed (e.g., for logging or checkpointing)
    global_rank = int(os.environ.get("RANK", 0))
    
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

    # Build the trainer and start training
    trainer = PPOTrainer(config)

    if args.eval:
        trainer.eval()
    else:
        trainer.train()
