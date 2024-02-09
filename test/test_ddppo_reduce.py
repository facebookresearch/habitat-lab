#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gc

import numpy as np
import pytest

from habitat.tasks.nav.nav import IntegratedPointGoalGPSAndCompassSensor

torch = pytest.importorskip("torch")
habitat_baselines = pytest.importorskip("habitat_baselines")

import gym
from torch import distributed as distrib
from torch import nn

from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.config.default import get_config
from habitat_baselines.rl.ddppo.algo import DDPPO
from habitat_baselines.rl.ddppo.ddp_utils import find_free_port
from habitat_baselines.rl.ppo.policy import PointNavBaselinePolicy


def _worker_fn(
    world_rank: int, world_size: int, port: int, unused_params: bool
):
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    tcp_store = distrib.TCPStore(  # type: ignore
        "127.0.0.1", port, world_size, world_rank == 0
    )
    distrib.init_process_group(
        "gloo", store=tcp_store, rank=world_rank, world_size=world_size
    )

    config = get_config("test/config/habitat_baselines/ppo_pointnav_test.yaml")
    obs_space = gym.spaces.Dict(
        {
            IntegratedPointGoalGPSAndCompassSensor.cls_uuid: gym.spaces.Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,),
                dtype=np.float32,
            )
        }
    )
    action_space = gym.spaces.Discrete(1)
    actor_critic = PointNavBaselinePolicy.from_config(
        config, obs_space, action_space
    )
    # This use adds some arbitrary parameters that aren't part of the computation
    # graph, so they will mess up DDP if they aren't correctly ignored by it
    if unused_params:
        actor_critic.unused = nn.Linear(64, 64)

    actor_critic.to(device=device)
    ppo_cfg = config.habitat_baselines.rl.ppo
    agent = DDPPO(
        actor_critic=actor_critic,
        clip_param=ppo_cfg.clip_param,
        ppo_epoch=ppo_cfg.ppo_epoch,
        num_mini_batch=ppo_cfg.num_mini_batch,
        value_loss_coef=ppo_cfg.value_loss_coef,
        entropy_coef=ppo_cfg.entropy_coef,
        lr=ppo_cfg.lr,
        eps=ppo_cfg.eps,
        max_grad_norm=ppo_cfg.max_grad_norm,
        use_normalized_advantage=ppo_cfg.use_normalized_advantage,
    )
    agent.init_distributed(find_unused_params=unused_params)
    rollouts = RolloutStorage(
        ppo_cfg.num_steps,
        2,
        obs_space,
        action_space,
        actor_critic,
        is_double_buffered=False,
    )
    rollouts.to(device)

    for k, v in rollouts.buffers["observations"].items():
        rollouts.buffers["observations"][k] = torch.randn_like(v)

    # Add two steps so batching works
    rollouts.advance_rollout()
    rollouts.advance_rollout()

    # Get a single batch
    batch = next(rollouts.data_generator(rollouts.buffers["returns"], 1))

    # Call eval actions through the internal wrapper that is used in
    # agent.update
    value, action_log_probs, dist_entropy, _, _ = agent._evaluate_actions(
        batch["observations"],
        batch["recurrent_hidden_states"],
        batch["prev_actions"],
        batch["masks"],
        batch["actions"],
        batch["rnn_build_seq_info"],
    )
    # Backprop on things
    (value.mean() + action_log_probs.mean() + dist_entropy.mean()).backward()

    # Make sure all ranks have very similar gradients
    for param in actor_critic.parameters():
        if param.grad is not None:
            grads = [param.grad.detach().clone() for _ in range(world_size)]
            distrib.all_gather(grads, grads[world_rank])

            for i in range(world_size):
                assert torch.isclose(grads[i], grads[world_rank]).all()

    torch.distributed.destroy_process_group()
    tcp_store = None
    gc.collect()


@pytest.mark.parametrize("unused_params", [True, False])
def test_ddppo_reduce(unused_params: bool):
    world_size = 2
    torch.multiprocessing.spawn(
        _worker_fn,
        args=(world_size, find_free_port(), unused_params),
        nprocs=world_size,
    )
