#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.storage import Storage
from habitat_baselines.common.tensor_dict import DictTree, TensorDict
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_pack_info_from_dones,
    build_rnn_build_seq_info,
)
from habitat_baselines.utils.common import get_action_space_info
from habitat_baselines.utils.timing import g_timer


@baseline_registry.register_storage
class RolloutStorage(Storage):
    r"""Class for storing rollout information for RL trainers."""

    def __init__(
        self,
        numsteps,
        num_envs,
        observation_space,
        action_space,
        actor_critic,
        is_double_buffered: bool = False,
    ):
        action_shape, discrete_actions = get_action_space_info(action_space)

        self.buffers = TensorDict()
        self.buffers["observations"] = TensorDict()

        for sensor in observation_space.spaces:
            self.buffers["observations"][sensor] = torch.from_numpy(
                np.zeros(
                    (
                        numsteps + 1,
                        num_envs,
                        *observation_space.spaces[sensor].shape,
                    ),
                    dtype=observation_space.spaces[sensor].dtype,
                )
            )

        self.buffers["recurrent_hidden_states"] = torch.zeros(
            numsteps + 1,
            num_envs,
            actor_critic.num_recurrent_layers,
            actor_critic.recurrent_hidden_size,
        )

        self.buffers["rewards"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["value_preds"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["returns"] = torch.zeros(numsteps + 1, num_envs, 1)

        self.buffers["action_log_probs"] = torch.zeros(
            numsteps + 1, num_envs, 1
        )

        if action_shape is None:
            action_shape = action_space.shape

        self.buffers["actions"] = torch.zeros(
            numsteps + 1, num_envs, *action_shape
        )
        self.buffers["prev_actions"] = torch.zeros(
            numsteps + 1, num_envs, *action_shape
        )
        if discrete_actions:
            assert isinstance(self.buffers["actions"], torch.Tensor)
            assert isinstance(self.buffers["prev_actions"], torch.Tensor)
            self.buffers["actions"] = self.buffers["actions"].long()
            self.buffers["prev_actions"] = self.buffers["prev_actions"].long()

        self.buffers["masks"] = torch.zeros(
            numsteps + 1, num_envs, 1, dtype=torch.bool
        )

        self.is_double_buffered = is_double_buffered
        self._nbuffers = 2 if is_double_buffered else 1
        self._num_envs = num_envs

        assert (self._num_envs % self._nbuffers) == 0

        self.num_steps = numsteps
        self.current_rollout_step_idxs = [0 for _ in range(self._nbuffers)]

        # The default device to torch is the CPU, so everything is on the CPU.
        self.device = torch.device("cpu")

    @property
    def current_rollout_step_idx(self) -> int:
        assert all(
            s == self.current_rollout_step_idxs[0]
            for s in self.current_rollout_step_idxs
        )
        return self.current_rollout_step_idxs[0]

    def to(self, device):
        self.buffers.map_in_place(lambda v: v.to(device))
        self.device = device

    @g_timer.avg_time("rollout_storage.insert", level=1)
    def insert(
        self,
        next_observations=None,
        next_recurrent_hidden_states=None,
        actions=None,
        action_log_probs=None,
        value_preds=None,
        rewards=None,
        next_masks=None,
        buffer_index: int = 0,
        **kwargs,
    ):
        if not self.is_double_buffered:
            assert buffer_index == 0

        next_step = dict(
            observations=next_observations,
            recurrent_hidden_states=next_recurrent_hidden_states,
            prev_actions=actions,
            masks=next_masks,
        )

        current_step = dict(
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=value_preds,
            rewards=rewards,
        )

        next_step = {k: v for k, v in next_step.items() if v is not None}
        current_step = {k: v for k, v in current_step.items() if v is not None}

        env_slice = slice(
            int(buffer_index * self._num_envs / self._nbuffers),
            int((buffer_index + 1) * self._num_envs / self._nbuffers),
        )

        if len(next_step) > 0:
            self.buffers.set(
                (self.current_rollout_step_idxs[buffer_index] + 1, env_slice),
                next_step,
                strict=False,
            )

        if len(current_step) > 0:
            self.buffers.set(
                (self.current_rollout_step_idxs[buffer_index], env_slice),
                current_step,
                strict=False,
            )

    def advance_rollout(self, buffer_index: int = 0):
        self.current_rollout_step_idxs[buffer_index] += 1

    def after_update(self):
        self.buffers[0] = self.buffers[self.current_rollout_step_idx]

        self.current_rollout_step_idxs = [
            0 for _ in self.current_rollout_step_idxs
        ]

    @g_timer.avg_time("rollout_storage.compute_returns", level=1)
    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            assert isinstance(self.buffers["value_preds"], torch.Tensor)
            self.buffers["value_preds"][
                self.current_rollout_step_idx
            ] = next_value
            gae = 0.0
            for step in reversed(range(self.current_rollout_step_idx)):
                delta = (
                    self.buffers["rewards"][step]
                    + gamma
                    * self.buffers["value_preds"][step + 1]
                    * self.buffers["masks"][step + 1]
                    - self.buffers["value_preds"][step]
                )
                gae = (
                    delta + gamma * tau * gae * self.buffers["masks"][step + 1]
                )
                self.buffers["returns"][step] = (  # type: ignore
                    gae + self.buffers["value_preds"][step]  # type: ignore
                )

        else:
            self.buffers["returns"][self.current_rollout_step_idx] = next_value
            for step in reversed(range(self.current_rollout_step_idx)):
                self.buffers["returns"][step] = (
                    gamma
                    * self.buffers["returns"][step + 1]
                    * self.buffers["masks"][step + 1]
                    + self.buffers["rewards"][step]
                )

    def data_generator(
        self,
        advantages: Optional[torch.Tensor],
        num_mini_batch: int,
    ) -> Iterator[DictTree]:
        assert isinstance(self.buffers["returns"], torch.Tensor)
        num_environments = self.buffers["returns"].size(1)
        assert num_environments >= num_mini_batch, (
            "Trainer requires the number of environments ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(
                num_environments, num_mini_batch
            )
        )
        if num_environments % num_mini_batch != 0:
            warnings.warn(
                "Number of environments ({}) is not a multiple of the"
                " number of mini batches ({}).  This results in mini batches"
                " of different sizes, which can harm training performance.".format(
                    num_environments, num_mini_batch
                )
            )

        dones_cpu = (
            torch.logical_not(self.buffers["masks"])
            .cpu()
            .view(-1, self._num_envs)
            .numpy()
        )
        for inds in torch.randperm(num_environments).chunk(num_mini_batch):
            curr_slice = (slice(0, self.current_rollout_step_idx), inds)

            batch = self.buffers[curr_slice]
            if advantages is not None:
                batch["advantages"] = advantages[curr_slice]
            batch["recurrent_hidden_states"] = batch[
                "recurrent_hidden_states"
            ][0:1]

            batch.map_in_place(lambda v: v.flatten(0, 1))

            batch["rnn_build_seq_info"] = build_rnn_build_seq_info(
                device=self.device,
                build_fn_result=build_pack_info_from_dones(
                    dones_cpu[
                        0 : self.current_rollout_step_idx, inds.numpy()
                    ].reshape(-1, len(inds)),
                ),
            )

            yield batch.to_tree()

    def __getstate__(self) -> Dict[str, Any]:
        return self.__dict__

    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__.update(state)

    def insert_first_observations(self, batch):
        self.buffers["observations"][0] = batch  # type: ignore

    def get_current_step(self, env_slice, buffer_index):
        return self.buffers[
            self.current_rollout_step_idxs[buffer_index],
            env_slice,
        ]

    def get_last_step(self):
        return self.buffers[self.current_rollout_step_idx]
