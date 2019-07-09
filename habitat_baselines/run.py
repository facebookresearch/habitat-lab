from habitat_baselines.common.base_model import get_model
from habitat_baselines.common.utils import experiment_args, get_exp_config


def main():
    parser = experiment_args()
    args = parser.parse_args()
    config = get_exp_config(args.exp_config)

    trainer = get_model(config.BASELINE.TRAINER_NAME, config)
    if args.run_type == "train":
        trainer.train()
    elif args.run_type == "eval":
        trainer.eval()


if __name__ == "__main__":
    main()
