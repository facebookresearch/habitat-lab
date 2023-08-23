import argparse
import pickle

import habitat
from habitat.datasets.rearrange.rearrange_dataset import RearrangeDatasetV0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-in-path",
        type=str,
        required=True,
        help="Path to the V1 dataset in .json.gz format",
    )
    parser.add_argument(
        "--dataset-out-path",
        type=str,
        required=True,
        help="Path to the V2 dataset in .pickle format",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="Habitat config task (something under `/benchmarks/`)",
    )
    args = parser.parse_args()

    if not args.dataset_in_path.endswith(".json.gz"):
        raise ValueError(
            "--dataset-in-path must specify a v1 dataset (ends with .json.gz)"
        )
    config = habitat.get_config(
        args.cfg, [f"habitat.dataset.data_path='{args.dataset_in_path}'"]
    )
    dataset = RearrangeDatasetV0(config.habitat.dataset)

    if not args.dataset_out_path.endswith(".pickle"):
        raise ValueError(
            "--dataset-out-path must specify a v2 dataset (ends with .pickle)"
        )

    with open(args.dataset_out_path, "wb") as f:
        pickle.dump(dataset.to_binary(), f)


if __name__ == "__main__":
    main()
