#!/usr/bin/env python3
import random
import sys
from typing import TYPE_CHECKING

import hydra
import numpy as np
import torch

from habitat.config.default import patch_config
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesConfigPlugin,
)
from rl_distance_train.imitation_learning.train import run_bc
from rl_distance_train import distance_policy

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="imitation_learning",
)
def main(cfg: "DictConfig"):
    # merge in habitat & habitat_baselines defaults
    cfg = patch_config(cfg)

    # reproducibility
    random.seed(cfg.habitat.seed)
    np.random.seed(cfg.habitat.seed)
    torch.manual_seed(cfg.habitat.seed)
    if cfg.habitat_baselines.force_torch_single_threaded and torch.cuda.is_available():
        torch.set_num_threads(1)

    # run behavioral cloning train
    run_bc(cfg)


if __name__ == "__main__":
    register_hydra_plugin(HabitatBaselinesConfigPlugin)
    main()
