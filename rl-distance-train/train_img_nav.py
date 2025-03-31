import random
import numpy as np
import torch
import os

from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_baselines.config.default import get_config
from omegaconf import OmegaConf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# A function to build configuration for PPO training
def build_PPO_config():
    config = get_config(os.path.join(SCRIPT_DIR, "configs", "ddppo_instance_imagenav.yaml"))

    OmegaConf.set_readonly(config, False)
    config.habitat_baselines.checkpoint_folder = "models/PPO_simple_dist_checkpoints"
    config.habitat_baselines.tensorboard_dir = "tb/PPO"
    # config.habitat_baselines.num_updates = -1
    config.habitat_baselines.verbose = True
    # config.habitat_baselines.num_checkpoints = -1
    # config.habitat_baselines.checkpoint_interval = 1000000
    # config.habitat_baselines.total_num_steps = 350 * 1000
    # del config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
    # del config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
    # config.habitat.dataset.data_path="data/datasets/pointnav/hm3d/v2/{split}/{split}.json.gz"
    OmegaConf.set_readonly(config, True)

    return config

if __name__ == "__main__":
    config = build_PPO_config()  # Build the config for PPO
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

    # Build the trainer and start training
    trainer = PPOTrainer(config)
    trainer.train()