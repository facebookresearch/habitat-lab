#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from scripts.utils import *
import numpy as np


# --- MCMC Sampler ---
def mcmc_rpy_constrained_sampling(
    q_init,
    rpy_target_deg,
    epsilon_deg=5.0,
    epsilon_pos=10.0,
    sigma=0.05,
    n_samples=500

):
    q_current = q_init.copy()
    samples = []

    samples.append(q_current.copy())

    if rpy_target_deg is None:
        rpy_target_deg = get_ee_rpy_deg(q_current)

    for _ in range(n_samples):
        q_proposal = q_current + np.random.normal(0, sigma, size=q_current.shape)

        # Check joint limits
        if np.any(q_proposal < joint_mins) or np.any(q_proposal > joint_maxs):
            samples.append(q_current.copy())
            continue

        # Check orientation constraint
        rpy_proposal = get_ee_rpy_deg(q_proposal)
        pos_proposal = get_ee_position(q_proposal)
        if rpy_distance_deg(rpy_proposal, rpy_target_deg) <= epsilon_deg and \
           np.linalg.norm(pos_proposal - get_ee_position(q_init)) <= epsilon_pos:
            q_current = q_proposal  # Accept
            samples.append(q_current.copy())         


        
    return np.array(samples)


def mcmc_callback(q_seed):

    
    mcmc_samples = mcmc_rpy_constrained_sampling(q_init= q_seed,
                                                    rpy_target_deg=None,
                                                    epsilon_deg=5.0,
                                                    epsilon_pos=0.1,
                                                    sigma=0.05,
                                                    n_samples=8000
                                                    )
    positions = []

    for q in mcmc_samples:
        positions.append(get_ee_position(q)-get_ee_position(q_seed))

    return get_basis_vectors_lengths(positions)


