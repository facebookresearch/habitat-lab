#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Tuple

import numpy as np


def compare_pose_trajectories(t1: np.ndarray, t2: np.ndarray):
    """
    Compares two numpy arrays containing a trajectory of joint configuration poses.
    """
    t1_shape: Tuple[int, ...] = t1.shape
    t2_shape: Tuple[int, ...] = t2.shape
    print(f"t1: {t1_shape}")
    print(f"t2: {t2_shape}")
    assert (
        t1_shape[1] == t2_shape[1]
    ), "Trajectory arrays must have the same shaped poses."
    if t1_shape[0] != t2_shape[0]:
        min_len = min(t1_shape[0], t2_shape[0])
        t1 = t1[:min_len]
        t2 = t2[:min_len]
    errors = np.linalg.norm(t1 - t2, axis=1)
    max_error: float = np.max(errors)
    max_error_idx = np.argmax(errors)
    print(f"errors = {errors}")
    print(f"max error {max_error} at index {max_error_idx} = {t1[0]-t2[0]}")
    print(f"frame 0 error = {t1[0]-t2[0]} -> {np.linalg.norm(t1[0] - t2[0])}")
    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Read and compare two numpy .npy files containing trajectories of robot poses."
    )

    parser.add_argument("--t1", type=str)
    parser.add_argument("--t2", type=str)

    args = parser.parse_args()

    # load the arrays

    t1 = np.load(args.t1)
    t2 = np.load(args.t2)

    compare_pose_trajectories(t1, t2)
