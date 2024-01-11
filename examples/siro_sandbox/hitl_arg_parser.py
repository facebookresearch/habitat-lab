#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Final

DEFAULT_CFG: Final[
    str
] = "experiments_hab3/pop_play_kinematic_oracle_humanoid_spot.yaml"


def create_hitl_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg", type=str, default=DEFAULT_CFG)
    parser.add_argument(
        "--cfg-opts",
        nargs="*",
        default=list(),
        help="Modify config options from command line",
    )

    return parser


def get_hitl_args():
    args = create_hitl_arg_parser().parse_args()

    return args
