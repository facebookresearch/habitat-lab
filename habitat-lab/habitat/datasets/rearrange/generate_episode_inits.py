#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

from tqdm import tqdm

import habitat


def generate_inits(cfg_path, opts):
    config = habitat.get_config(cfg_path, opts)
    with habitat.Env(config=config) as env:
        for i in tqdm(range(env.number_of_episodes)):
            if i % 100 == 0:
                # Print the dataset we are generating initializations for. This
                # is useful when this script runs for a long time and we don't
                # know which dataset the job is for.
                print(cfg_path, config.habitat.dataset.data_path)
            env.reset()


parser = argparse.ArgumentParser()
parser.add_argument("--cfg-path", type=str, required=True)
parser.add_argument(
    "opts",
    default=None,
    nargs=argparse.REMAINDER,
    help="Modify config options from command line",
)
args = parser.parse_args()


generate_inits(args.cfg_path, args.opts)
