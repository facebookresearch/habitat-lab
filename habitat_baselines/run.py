import argparse
import random
from typing import List

import numpy as np

from habitat import Config, get_config
from habitat_baselines.common.utils import get_trainer
from habitat_baselines.config.default import get_config as baseline_cfg


def get_exp_config(cfg_path: str, opts: List[str] = None) -> Config:
    r"""
    Create config object from path for a specific experiment run.
    Args:
        cfg_path: yaml config file path.
        opts: list additional options or options to be overwritten.

    Returns:
        config object created.
    """

    config = Config(new_allowed=True)
    config.merge_from_other_cfg(baseline_cfg(cfg_path))
    task_config = get_config(config.BASE_TASK_CONFIG_PATH)
    config.TASK_CONFIG = task_config
    config.CMD_TRAILING_OPTS = opts
    if opts is not None:
        config.merge_from_list(opts)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()
    config = get_exp_config(args.exp_config, args.opts)

    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)

    trainer = get_trainer(config.TRAINER.TRAINER_NAME, config)
    if args.run_type == "train":
        trainer.train()
    elif args.run_type == "eval":
        trainer.eval()


if __name__ == "__main__":
    main()
