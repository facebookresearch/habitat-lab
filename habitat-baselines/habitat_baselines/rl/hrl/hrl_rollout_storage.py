#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterator, Optional

import torch

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensor_dict import DictTree, TensorDict
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_pack_info_from_dones,
    build_rnn_build_seq_info,
)

EPS_PPO = 1e-5


@baseline_registry.register_storage
class HrlRolloutStorage(RolloutStorage):
    """
    Supports variable writes to the rollout buffer where data is not inserted
    into the buffer on every step. When getting batches from the storage, these
    batches will only contain samples that were written. This means that the
    batches could be variable size and less than the maximum size of the
    rollout buffer.
    """

    def __init__(self, numsteps, num_envs, *args, **kwargs):
        super().__init__(numsteps, num_envs, *args, **kwargs)
        self._num_envs = num_envs
        self._cur_step_idxs = torch.zeros(self._num_envs, dtype=torch.long)
        self._last_should_inserts = None
        self._current_step = {}
        assert (
            not self.is_double_buffered
        ), "HRL storage does not support double buffered sampling"

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
        should_inserts: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):
        """
        The only key different from the base `RolloutStorage` is
        `should_inserts`. This is a bool tensor of shape [# environments,]. If
        `should_insert[i] == True`, then this will the sample at enviroment
        index `i` into the rollout buffer at environment index `i`, if not, it
        will ignore the sample. If None, this defaults to the last insert
        state.

        Rewards acquired of steps where `should_insert[i] == False` will be summed up and added to the next step where `should_insert[i] == True`
        """

        # The actions here could be Float instead of long because we previously
        # concatenated them with actions from other agents that have low level policies.
        if (
            type(self.buffers["actions"]) is torch.Tensor
            and actions is not None
        ):
            actions = actions.type(self.buffers["actions"].dtype)

        if next_masks is not None:
            next_masks = next_masks.to(self.device)
        if rewards is not None:
            rewards = rewards.to(self.device)
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
        )

        next_step = TensorDict(
            {k: v for k, v in next_step.items() if v is not None}
        )
        current_step = TensorDict(
            {k: v for k, v in current_step.items() if v is not None}
        )
        self._current_step.update(next_step)

        if should_inserts is None:
            should_inserts = self._last_should_inserts
        assert should_inserts is not None
        # Starts as shape [batch_size, 1]
        should_inserts = should_inserts.flatten()

        if should_inserts.sum() == 0:
            self._last_should_inserts = should_inserts
            return

        env_idxs = torch.arange(self._num_envs)
        if rewards is not None:
            rewards = rewards.to(self.device)
            # Accumulate rewards between writes to the observations.
            self.buffers["rewards"][self._cur_step_idxs, env_idxs] += rewards

        if len(next_step) > 0:
            self.buffers.set(
                (
                    self._cur_step_idxs[should_inserts] + 1,
                    env_idxs[should_inserts],
                ),
                next_step[should_inserts],
                strict=False,
            )

        if len(current_step) > 0:
            self.buffers.set(
                (
                    self._cur_step_idxs[should_inserts],
                    env_idxs[should_inserts],
                ),
                current_step[should_inserts],
                strict=False,
            )
        self._last_should_inserts = should_inserts

    def advance_rollout(self, buffer_index: int = 0):
        """
        This will advance to writing at the next step in the data buffer ONLY
        if an element was written to that environment index in the previous
        step.
        """
        self._cur_step_idxs += self._last_should_inserts

    def after_update(self):
        env_idxs = torch.arange(self._num_envs)
        self.buffers[0] = self.buffers[self._cur_step_idxs, env_idxs]
        self.buffers["masks"][1:] = False
        self.buffers["rewards"][1:] = 0.0

        self._cur_step_idxs[:] = 0

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if not use_gae:
            raise ValueError("Only GAE is supported with HRL trainer")

        assert isinstance(self.buffers["value_preds"], torch.Tensor)
        gae = 0.0
        for step in reversed(range(self._cur_step_idxs.max())):
            delta = (
                self.buffers["rewards"][step]
                + gamma
                * self.buffers["value_preds"][step + 1]
                * self.buffers["masks"][step + 1]
                - self.buffers["value_preds"][step]
            )
            gae = delta + gamma * tau * gae * self.buffers["masks"][step + 1]
            self.buffers["returns"][step] = (  # type: ignore
                gae + self.buffers["value_preds"][step]  # type: ignore
            )

    def data_generator(self, advantages, num_batches) -> Iterator[DictTree]:
        """
        Generates data batches based on the data that has been written to the
        rollout buffer.
        """

        num_environments = advantages.size(1)
        dones_cpu = (
            torch.logical_not(self.buffers["masks"])
            .cpu()
            .view(-1, self._num_envs)
            .numpy()
        )
        for inds in torch.randperm(num_environments).chunk(num_batches):
            batch = self.buffers[0 : self.num_steps, inds]
            batch["advantages"] = advantages[: self.num_steps, inds]
            batch["recurrent_hidden_states"] = batch[
                "recurrent_hidden_states"
            ][0:1]
            batch["loss_mask"] = (
                torch.arange(self.num_steps, device=advantages.device)
                .view(-1, 1, 1)
                .repeat(1, len(inds), 1)
            )
            for i, env_i in enumerate(inds):
                # Stricly less than is so we throw out the last transition. We
                # need to throw out the last transition because we were not
                # able to accumulate rewards for it.
                batch["loss_mask"][:, i] = batch["loss_mask"][:, i] < (
                    self._cur_step_idxs[env_i] - 1
                )

            batch.map_in_place(lambda v: v.flatten(0, 1))
            batch["rnn_build_seq_info"] = build_rnn_build_seq_info(
                device=self.device,
                build_fn_result=build_pack_info_from_dones(
                    dones_cpu[0 : self.num_steps, inds.numpy()].reshape(
                        -1, len(inds)
                    ),
                ),
            )

            yield batch.to_tree()

    @property
    def current_rollout_step_idxs(self):
        # To ensure we aren't accessing this property from the base rollout
        # storage and separately tracking the current write index.
        raise ValueError()

    @current_rollout_step_idxs.setter
    def current_rollout_step_idxs(self, val):
        pass

    @property
    def current_rollout_step_idx(self):
        # To ensure we aren't accessing this property from the base rollout
        # storage and separately tracking the current write index.
        raise ValueError()

    def get_current_step(self, env_slice, buffer_index):
        return TensorDict(self._current_step)

    def insert_first_observations(self, batch):
        super().insert_first_observations(batch)
        self._current_step = self.buffers[0]

    def get_last_step(self):
        env_idxs = torch.arange(self._num_envs)
        return self.buffers[self._cur_step_idxs, env_idxs]
