#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterator

import torch

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensor_dict import TensorDict

EPS_PPO = 1e-5


@baseline_registry.register_policy
class HrlRolloutStorage(RolloutStorage):
    def __init__(self, numsteps, num_envs, *args, **kwargs):
        super().__init__(numsteps, num_envs, *args, **kwargs)
        self._num_envs = num_envs
        self._numsteps = numsteps
        self._cur_step_idxs = torch.zeros(self._num_envs, dtype=torch.long)
        self._last_should_inserts = None
        if self.is_double_buffered:
            raise ValueError(
                "HRL storage does not support double buffered sampling"
            )

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
        should_inserts=None,
    ):
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

        next_step = {k: v for k, v in next_step.items() if v is not None}
        current_step = {k: v for k, v in current_step.items() if v is not None}

        if should_inserts is None:
            should_inserts = self._last_should_inserts
        assert should_inserts is not None

        if should_inserts.sum() == 0:
            return

        env_idxs = torch.arange(self._num_envs)
        if rewards is not None:
            # Accumulate rewards between updates.
            reward_write_idxs = torch.maximum(
                self._cur_step_idxs - 1, torch.zeros_like(self._cur_step_idxs)
            )
            self.buffers["rewards"][reward_write_idxs, env_idxs] += rewards.to(
                self.device
            )

        if len(next_step) > 0:
            self.buffers.set(
                (
                    self._cur_step_idxs[should_inserts] + 1,
                    env_idxs[should_inserts],
                ),
                next_step,
                strict=False,
            )

        if len(current_step) > 0:
            self.buffers.set(
                (
                    self._cur_step_idxs[should_inserts],
                    env_idxs[should_inserts],
                ),
                current_step,
                strict=False,
            )
        self._last_should_inserts = should_inserts

    def advance_rollout(self, buffer_index: int = 0):
        self._cur_step_idxs += self._last_should_inserts.long()

        is_past_buffer = self._cur_step_idxs >= self._numsteps
        if is_past_buffer.sum() > 0:
            self._cur_step_idxs[is_past_buffer] = self._numsteps - 1
            env_idxs = torch.arange(self._num_envs)
            self.buffers["rewards"][
                self._cur_step_idxs[is_past_buffer], env_idxs[is_past_buffer]
            ] = 0.0

    def after_update(self):
        env_idxs = torch.arange(self._num_envs)
        self.buffers[0] = self.buffers[self._cur_step_idxs, env_idxs]
        self.buffers["masks"][1:] = False
        self.buffers["rewards"][1:] = 0.0

        self.current_rollout_step_idxs = [
            0 for _ in self.current_rollout_step_idxs
        ]
        self._cur_step_idxs[:] = 0

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if not use_gae:
            raise ValueError("Only GAE is supported with HRL trainer")

        assert isinstance(self.buffers["value_preds"], torch.Tensor)
        gae = 0.0
        for step in reversed(range(self._cur_step_idxs.max() - 1)):
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

    def recurrent_generator(
        self, advantages, num_batches
    ) -> Iterator[TensorDict]:
        num_environments = advantages.size(1)
        for inds in torch.randperm(num_environments).chunk(num_batches):
            batch = self.buffers[0 : self._numsteps, inds]
            batch["advantages"] = advantages[: self._numsteps, inds]
            batch["recurrent_hidden_states"] = batch[
                "recurrent_hidden_states"
            ][0:1]
            batch["loss_mask"] = (
                torch.arange(self._numsteps, device=advantages.device)
                .view(-1, 1, 1)
                .repeat(1, len(inds), 1)
            )
            for i, env_i in enumerate(inds):
                # The -1 is to throw out the last transition.
                batch["loss_mask"][:, i] = (
                    batch["loss_mask"][:, i] < self._cur_step_idxs[env_i] - 1
                )

            yield batch.map(lambda v: v.flatten(0, 1))
