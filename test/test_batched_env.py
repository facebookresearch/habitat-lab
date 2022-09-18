#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

input("press enter to begin...")  # needed for py-spy capture

import pickle
import time
from collections import OrderedDict

import torch

import habitat_sim
from habitat.core.batched_env import BatchedEnv
from habitat_baselines.config.default import get_config

next_episode = 0


def are_floats_equal(a, b, eps):
    assert type(a) == type(b)
    if isinstance(a, float):
        return abs(a - b) < eps
    elif isinstance(a, (str, int)):
        return a == b
    elif isinstance(a, (list, tuple)):
        return all(
            are_floats_equal(item_a, item_b, eps)
            for item_a, item_b in zip(a, b)
        )
    elif isinstance(a, (OrderedDict, dict)):
        return all(
            are_floats_equal(item_a, item_b, eps)
            for item_a, item_b in zip(a.items(), b.items())
        )
    else:
        try:
            _ = iter(a)
        except TypeError:
            # not iterable
            return a == b
        else:
            raise AssertionError()


def test_basic():

    pickle_filepath = "test_batched_env_test_basic_outputs_by_step.pickle"

    try:
        with open(pickle_filepath, "rb") as f:
            ref_outputs_by_step = pickle.load(f)
    except FileNotFoundError:
        ref_outputs_by_step = None

    exp_config = (
        "habitat_baselines/config/rearrange/gala_kinematic_local_small.yaml"
    )
    opts = None
    config = get_config(exp_config, opts)

    batched_env = BatchedEnv(config)

    _ = batched_env.reset()

    outputs_by_step = []

    for _ in range(10):
        # actions = torch.rand(batched_env.num_envs, batched_env.action_dim)
        actions = torch.ones(batched_env.num_envs, batched_env.action_dim)
        batched_env.async_step(actions)
        outputs = batched_env.wait_step()

        # convert tensors to lists
        batched_observations, _, _, _ = outputs
        for key in batched_observations.keys():
            value = batched_observations[key]
            if isinstance(value, torch.torch.Tensor):
                batched_observations[key] = value.tolist()
        outputs_by_step.append(outputs)

    if ref_outputs_by_step:
        assert are_floats_equal(ref_outputs_by_step, outputs_by_step, 1e-6)
        print("success")
    else:
        with open(pickle_filepath, "wb") as f:
            pickle.dump(outputs_by_step, f)
        print("wrote ", pickle_filepath)

    # batched_env.close()


def test_benchmark():

    print("build_type: ", habitat_sim.build_type)

    exp_config = "habitat_baselines/config/rearrange/gala_kinematic_ddppo.yaml"
    exp_config_local = (
        "habitat_baselines/config/rearrange/gala_kinematic_local.yaml"
    )
    opts = None
    config = get_config([exp_config, exp_config_local], opts)

    batched_env = BatchedEnv(config)

    _ = batched_env.reset()

    torch.manual_seed(0)

    t_recent = time.time()

    for batch_step_idx in range(4000):
        actions = torch.rand(batched_env.num_envs, batched_env.action_dim)
        batched_env.async_step(actions)
        # todo: use outputs to ensure they don't get optimized away
        _ = batched_env.wait_step()

        log_interval = 100
        if batch_step_idx != 0 and batch_step_idx % log_interval == 0:
            t_curr = time.time()
            num_recent_steps = log_interval * batched_env.num_envs
            sps = num_recent_steps / (t_curr - t_recent)
            t_recent = t_curr

            print(
                "batch_step_idx: {}, sps: {:.2f}".format(batch_step_idx, sps)
            )


if __name__ == "__main__":
    # test_benchmark()
    test_basic()
