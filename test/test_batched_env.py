#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import random
import time
from collections import OrderedDict

import numpy as np
import torch

import habitat_sim
from habitat.core.batched_env import BatchedEnv
from habitat_baselines.config.default import get_config
from habitat_sim._ext.habitat_sim_bindings import PythonEnvironmentState

next_episode = 0


def are_scalars_equal(a, b, eps):
    assert type(a) == type(b)
    if isinstance(a, float):
        return abs(a - b) < eps
    elif isinstance(a, (str, int)):
        return a == b
    elif isinstance(a, (list, tuple, np.ndarray)):
        return all(
            are_scalars_equal(item_a, item_b, eps)
            for item_a, item_b in zip(a, b)
        )
    elif isinstance(a, (OrderedDict, dict)):
        return all(
            are_scalars_equal(item_a, item_b, eps)
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
        assert are_scalars_equal(ref_outputs_by_step, outputs_by_step, 1e-6)
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

    print("config.NUM_ENVIRONMENTS: ", config.NUM_ENVIRONMENTS)

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


class FakeBatchEnvironmentState:
    def __init__(self, num_envs):

        self.num_envs = num_envs
        self.episode_step_idx = np.zeros((num_envs), dtype=int)

        self.target_obj_idx = np.zeros((num_envs), dtype=int)
        self.held_obj_idx = np.full((num_envs), -1, dtype=int)
        self.did_drop = np.zeros((num_envs), dtype=bool)

        self.target_obj_start_pos = np.zeros((num_envs, 3), dtype=float)

        self.goal_pos = np.zeros((num_envs, 3), dtype=float)
        self.ee_pos = np.zeros((num_envs, 3), dtype=float)
        self.target_obj_pos = np.zeros((num_envs, 3), dtype=float)

    def to_env_state(self):
        env_state = [PythonEnvironmentState() for _ in range(self.num_envs)]

        for b in range(self.num_envs):
            env_state[b].episode_step_idx = self.episode_step_idx[b]
            env_state[b].target_obj_idx = self.target_obj_idx[b]
            env_state[b].held_obj_idx = self.held_obj_idx[b]
            env_state[b].target_obj_start_pos = self.target_obj_start_pos[b]
            env_state[b].goal_pos = self.goal_pos[b]
            env_state[b].ee_pos = self.ee_pos[b]
            assert env_state[b].target_obj_idx == 0
            env_state[b].obj_positions = [self.target_obj_pos[b]]
            env_state[b].did_drop = self.did_drop[b]

        return env_state


def test_get_batch_dones_rewards_resets(*args):

    exp_config = "habitat_baselines/config/rearrange/gala_kinematic_ddppo.yaml"
    exp_config_local = (
        "habitat_baselines/config/rearrange/gala_kinematic_local.yaml"
    )
    config = get_config([exp_config, exp_config_local], list(args))

    batched_env = BatchedEnv(config)

    _ = batched_env.reset()

    torch.manual_seed(0)
    random.seed(0)

    frozen_next_episode_idx = batched_env._next_episode_idx
    compare_eps = 1.0e-6

    prev_batch_env_state = FakeBatchEnvironmentState(batched_env._num_envs)

    def get_small_random_position():
        half_size = 0.2
        return [
            random.uniform(-half_size, half_size),
            random.uniform(-half_size, half_size),
            random.uniform(-half_size, half_size),
        ]

    dones_count = 0
    success_count = 0

    for i in range(5000):

        # actions = torch.tensor(np.zeros((batched_env._num_envs, batched_env.action_dim), dtype=float))
        actions = (
            torch.rand(batched_env.num_envs, batched_env.action_dim) * 2.0
            - 1.0
        )

        batch_env_state = FakeBatchEnvironmentState(batched_env._num_envs)
        s = batch_env_state
        ps = prev_batch_env_state
        for b in range(s.num_envs):

            if i == 0 or batched_env.dones[b]:
                s.episode_step_idx[b] = 0
                s.goal_pos[b] = get_small_random_position()
            else:
                s.episode_step_idx[b] = ps.episode_step_idx[b] + 1
                s.goal_pos[b] = ps.goal_pos[b]

            if s.episode_step_idx[b] == 0:
                s.held_obj_idx[b] = -1
                s.did_drop[b] = False
            else:
                s.held_obj_idx[b] = (
                    ps.held_obj_idx[b]
                    if random.random() < 0.9
                    else random.randint(-1, 1)
                )
                if s.held_obj_idx[b] == -1 and ps.held_obj_idx[b] != -1:
                    s.did_drop[b] = True
            s.target_obj_start_pos[b] = get_small_random_position()
            s.ee_pos[b] = get_small_random_position()
            s.target_obj_pos[b] = get_small_random_position()

        batched_env._next_episode_idx = frozen_next_episode_idx
        batched_env.get_batch_dones_rewards_resets_helper(
            batch_env_state, prev_batch_env_state, actions
        )
        rewards_copy = batched_env.rewards.copy()
        dones_copy = batched_env.dones.copy()
        resets_copy = batched_env.resets.copy()

        env_state = batch_env_state.to_env_state()
        # batched_env._previous_state = prev_batch_env_state.to_env_state()
        batched_env._next_episode_idx = frozen_next_episode_idx
        actions_flat_list = actions.flatten().tolist()
        batched_env.get_dones_rewards_resets(env_state, actions_flat_list)
        assert are_scalars_equal(
            rewards_copy, batched_env.rewards, compare_eps
        )
        assert are_scalars_equal(dones_copy, batched_env.dones, compare_eps)
        assert are_scalars_equal(resets_copy, batched_env.resets, compare_eps)

        dones_count += np.count_nonzero(dones_copy)
        success_count += np.count_nonzero(
            rewards_copy >= batched_env._config.NPNP_SUCCESS_REWARD
        )

        prev_batch_env_state = batch_env_state

    assert dones_count >= 10
    print(f"Success count {success_count}, Arguments : {list(args)}")
    assert success_count >= 10


if __name__ == "__main__":
    # test_benchmark()
    # test_basic()
    print("\n ##### \n")
    # test_get_batch_dones_rewards_resets()
    # test_get_batch_dones_rewards_resets("TASK_HAS_SIMPLE_PLACE", True)
    # test_get_batch_dones_rewards_resets("TASK_HAS_SIMPLE_PLACE", True, "DO_NOT_END_IF_DROP_WRONG", True)
    # test_get_batch_dones_rewards_resets("PREVENT_STOP_ACTION", True, "TASK_NO_END_ACTION", True)
    test_get_batch_dones_rewards_resets(
        "TASK_HAS_SIMPLE_PLACE",
        True,
        "DO_NOT_END_IF_DROP_WRONG",
        True,
        "PREVENT_STOP_ACTION",
        True,
        "TASK_NO_END_ACTION",
        True,
    )
    print("All tests passed")
