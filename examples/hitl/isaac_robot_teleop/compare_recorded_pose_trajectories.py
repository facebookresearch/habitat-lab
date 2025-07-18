#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

# this vector contains lower and upper joint limits for the hardware dof
# NOTE: used to scale the graph visualizations and computed in record_post_process.py
sim_limits_in_hardware_order = [
    [
        -2.743699789047241,
        -1.7836999893188477,
        -2.9006998538970947,
        -3.042099952697754,
        -2.80649995803833,
        0.544499933719635,
        -3.015899658203125,
        -0.4699999690055847,
        -0.19599997997283936,
        -0.17399999499320984,
        -0.22699998319149017,
        -0.4699999690055847,
        -0.19599997997283936,
        -0.17399999499320984,
        -0.22699998319149017,
        -0.4699999690055847,
        -0.19599997997283936,
        -0.17399999499320984,
        -0.22699998319149017,
        0.2629999816417694,
        -0.10499999672174454,
        -0.1889999806880951,
        -0.16199998557567596,
        -2.743699789047241,
        -1.7836999893188477,
        -2.9006998538970947,
        -3.042099952697754,
        -2.80649995803833,
        0.544499933719635,
        -3.015899658203125,
        -0.4699999690055847,
        -0.19599997997283936,
        -0.17399999499320984,
        -0.22699998319149017,
        -0.4699999690055847,
        -0.19599997997283936,
        -0.17399999499320984,
        -0.22699998319149017,
        -0.4699999690055847,
        -0.19599997997283936,
        -0.17399999499320984,
        -0.22699998319149017,
        0.2629999816417694,
        -0.10499999672174454,
        -0.1889999806880951,
        -0.16199998557567596,
    ],
    [
        2.743699789047241,
        1.7836999893188477,
        2.9006998538970947,
        -0.1517999917268753,
        2.80649995803833,
        4.516899585723877,
        3.015899658203125,
        0.4699999690055847,
        1.6099998950958252,
        1.7089998722076416,
        1.6179999113082886,
        0.4699999690055847,
        1.6099998950958252,
        1.7089998722076416,
        1.6179999113082886,
        0.4699999690055847,
        1.6099998950958252,
        1.7089998722076416,
        1.6179999113082886,
        1.3960000276565552,
        1.1629998683929443,
        1.6439999341964722,
        1.7189998626708984,
        2.743699789047241,
        1.7836999893188477,
        2.9006998538970947,
        -0.1517999917268753,
        2.80649995803833,
        4.516899585723877,
        3.015899658203125,
        0.4699999690055847,
        1.6099998950958252,
        1.7089998722076416,
        1.6179999113082886,
        0.4699999690055847,
        1.6099998950958252,
        1.7089998722076416,
        1.6179999113082886,
        0.4699999690055847,
        1.6099998950958252,
        1.7089998722076416,
        1.6179999113082886,
        1.3960000276565552,
        1.1629998683929443,
        1.6439999341964722,
        1.7189998626708984,
    ],
]


def compare_pose_trajectories(t1: np.ndarray, t2: np.ndarray):
    """
    Compares two numpy arrays containing a trajectory of joint configuration poses.
    Also plots each dimension of the pose trajectories.
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
    print(
        f"max error {max_error} at index {max_error_idx} = {t1[max_error_idx]-t2[max_error_idx]}"
    )
    print(f"frame 0 error = {t1[0]-t2[0]} -> {np.linalg.norm(t1[0] - t2[0])}")

    # Plot each dimension in a grid for better visualization
    num_dims = t1_shape[1]
    ncols = min(3, num_dims)
    nrows = int(np.ceil(num_dims / ncols))
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 3 * nrows), sharex=True
    )
    axs = np.array(axs).reshape(-1)  # Flatten in case of single row/col

    # Find global min and max across both trajectories
    # global_min: float = min(np.min(t1), np.min(t2))
    # global_max: float = max(np.max(t1), np.max(t2))

    ix_correspondance = [
        ("left arm", 6),
        ("left hand", 22),
        ("right arm", 29),
        ("right hand", 45),
    ]

    for i in range(num_dims):
        axs[i].plot(t1[:, i], label="real")
        axs[i].plot(t2[:, i], label="sim")
        association = "unknown"
        for part, max_ix in ix_correspondance:
            if max_ix >= i:
                association = part
                break
        axs[i].set_ylabel(
            f"{i} - {association}", rotation=0, ha="right", va="center"
        )
        axs[i].set_ylim(
            sim_limits_in_hardware_order[0][i],
            sim_limits_in_hardware_order[1][i],
        )
    axs[0].legend()
    for ax in axs[num_dims:]:
        ax.axis("off")
    axs[-1].set_xlabel("Frame")
    plt.tight_layout()
    plt.show()

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
