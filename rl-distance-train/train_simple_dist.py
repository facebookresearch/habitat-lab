import random
import numpy as np
import torch
import os

from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_baselines.config.default import get_config
from omegaconf import OmegaConf

# A function to build configuration for PPO training
def build_PPO_config(config_path="pointnav/ddppo_pointnav.yaml"):
    config = get_config(config_path)
    # Change for REINFORCE
    OmegaConf.set_readonly(config, False)
    config.habitat_baselines.checkpoint_folder = "rl-distance-train/PPO-dist-checkpoints/policy-${now:%Y-%m-%d}_${now:%H-%M-%S}"
    config.habitat_baselines.tensorboard_dir = "rl-distance-train/tb/PPO-dist-${now:%Y-%m-%d}_${now:%H-%M-%S}"
    config.habitat_baselines.rl.policy.main_agent.name = "PointNavGoalDistancePolicy"
    # del config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
    del config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
    # config.habitat.dataset.data_path="data/datasets/pointnav/simple_room/v0/{split}/empty_room.json.gz"
    OmegaConf.set_readonly(config, True)


    return config

if __name__ == "__main__":
    # Build the config for PPO
    config = build_PPO_config()

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
