import random

import numpy as np

from habitat_baselines.common.utils import (
    experiment_args,
    get_exp_config,
    get_trainer,
)


def main():
    parser = experiment_args()
    args = parser.parse_args()
    config = get_exp_config(args.exp_config)

    random.seed(config.SEED)
    np.random.seed(config.SEED)

    trainer = get_trainer(config.TRAINER.TRAINER_NAME, config)
    if args.run_type == "train":
        trainer.train()
    elif args.run_type == "eval":
        trainer.eval()


if __name__ == "__main__":
    main()
