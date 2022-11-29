#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.action_embedding import ActionEmbedding
from habitat_baselines.rl.ppo.policy import Net


def _unflatten(t: torch.Tensor, dim: int, sizes: torch.Size):
    t_size = list(t.size())
    unflatten_size = t_size[0:dim] + list(sizes) + t_size[dim + 1 :]

    return t.contiguous().view(unflatten_size)


def masked_mean(t, valids):
    r"""Compute the mean of t for the valid locations specified in valids"""
    assert valids.numel() > 0
    invalids = torch.logical_not(valids)
    t = torch.masked_fill(t, invalids, 0.0)

    return t.mean() * (valids.float().sum() / valids.numel())


def masked_index_select(
    t, indexer, valids, dim: int = 0, fill_value: float = 0.0
):
    r"""Selects the indices of t specified by indexer. Masks
    out invalid values and replaces them with fill_value.

    :param t: The tensor to index, shape (N, *)
    :param indexer: Indexing tensor any shape. The maximum valid value
    must be less than N and greater than -1.

    :param valids: The valid indices to select.
    :param dim: The indexing dim. zero by default. The docstring assumes
        this value is zero
    :param fill_value: The value to fill invalid locations

    :return: Tensor of shape (:ref:`indexer`, *)
    """
    indexer_size = indexer.size()
    not_valids = torch.logical_not(valids)

    indexer = torch.masked_fill(indexer, not_valids, 0).view(-1)
    output = t.index_select(dim, indexer)

    mask_size = [1 for _ in range(output.dim())]
    mask_size[dim] = not_valids.numel()
    output.masked_fill_(not_valids.view(mask_size), fill_value)

    return _unflatten(output, dim, indexer_size)


class ActionConditionedForwardModelingLoss(nn.Module):
    r"""Generic base for a action-conditioned forward modeling loss.

    Takes the output of the policy RNN and uses a secondary RNN
    to create future predictions based on the action sequence
    {a_t, ..., a_{t+k-1}}

    For each episode in the mini-batch, predictions are made for
    (up to) time_subsample timesteps. This is done to reduce
    gradient correlation.

    Also returned are (up to) future_subsample inds per future trajectory.
    This can be used to further reduce correlation. Note that
    subsampling must be done by the callee. The full future predictions
    are returned incase that is needed by the downstream loss.
    """

    def __init__(
        self,
        action_space: gym.spaces.Dict,
        hidden_size: int,
        k: int = 20,
        time_subsample: int = 6,
        future_subsample: int = 2,
    ):
        super().__init__()

        self._action_embed = ActionEmbedding(action_space)

        self._future_predictor = nn.LSTM(
            self._action_embed.output_size, hidden_size
        )

        self.k = k
        self.time_subsample = time_subsample
        self.future_subsample = future_subsample
        self._hidden_size = hidden_size

        self.layer_init()

    def layer_init(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def _build_inds(self, rnn_build_seq_info):
        num_seqs_at_step = rnn_build_seq_info["num_seqs_at_step"]
        sequence_lengths = rnn_build_seq_info["cpu_sequence_lengths"]
        device = num_seqs_at_step.device

        shortest_seq = sequence_lengths.min()
        longest_seq = sequence_lengths.max()
        n_seqs = sequence_lengths.size(0)
        shortest_seq, longest_seq = map(
            lambda t: t.item(), (shortest_seq, longest_seq)
        )

        start_times = []
        seq_offsets = []
        max_valid = []
        for i in range(n_seqs):
            if self.time_subsample >= (sequence_lengths[i] - 1):
                start_times.append(
                    torch.arange(
                        1,
                        sequence_lengths[i],
                        device=device,
                        dtype=torch.int64,
                    )
                )
            else:
                start_times.append(
                    torch.randperm(
                        sequence_lengths[i], device=device, dtype=torch.int64
                    )[0 : self.time_subsample]
                )

            seq_offsets.append(torch.full_like(start_times[-1], i))
            max_valid.append(
                torch.full_like(start_times[-1], sequence_lengths[i] - 1)
            )

        start_times = torch.cat(start_times, dim=0)
        seq_offsets = torch.cat(seq_offsets, dim=0)
        max_valid = torch.cat(max_valid, dim=0)

        all_times = torch.arange(
            self.k, dtype=torch.int64, device=device
        ).view(-1, 1) + start_times.view(1, -1)

        action_valids = all_times < max_valid.view(1, -1)
        target_valids = (all_times + 1) < max_valid.view(1, -1)
        all_times[torch.logical_not(action_valids)] = 0

        time_start_inds = torch.cumsum(num_seqs_at_step, 0) - num_seqs_at_step
        action_inds = time_start_inds[all_times] + seq_offsets.view(1, -1)
        target_inds = time_start_inds[all_times + 1] + seq_offsets.view(1, -1)

        select_inds = rnn_build_seq_info["select_inds"]

        action_inds, target_inds = map(
            lambda t: select_inds.index_select(0, t.flatten()).view_as(t),
            (action_inds, target_inds),
        )

        return action_inds, target_inds, action_valids, target_valids

    def forward(self, aux_loss_state, batch):
        action = self._action_embed(batch["action"])

        (
            action_inds,
            target_inds,
            action_valids,
            target_valids,
        ) = self._build_inds(batch["rnn_build_seq_info"])

        hidden_states = masked_index_select(
            aux_loss_state["rnn_output"], action_inds[0], action_valids[0]
        ).unsqueeze(0)
        action = masked_index_select(action, action_inds, action_valids)

        future_preds, _ = self._future_predictor(
            action, (hidden_states, hidden_states)
        )

        k = action.size(0)
        num_samples = self.future_subsample
        if num_samples < k:
            future_inds = torch.multinomial(
                action.new_full((), 1.0 / k).expand(action.size(1), k),
                num_samples=num_samples,
                replacement=False,
            )
        else:
            future_inds = (
                torch.arange(k, device=action.device, dtype=torch.int64)
                .view(k, 1)
                .expand(k, action.size(1))
            )

        future_inds = (
            future_inds
            + torch.arange(
                0,
                future_inds.size(0) * k,
                k,
                device=future_inds.device,
                dtype=future_inds.dtype,
            ).view(-1, 1)
        ).flatten()

        return (
            future_preds,
            action_inds,
            target_inds,
            future_inds,
            action_valids,
            target_valids,
        )


@baseline_registry.register_auxiliary_loss(name="cpca")
class CPCA(ActionConditionedForwardModelingLoss):
    r"""Implements Action-conditional Contrastive Predictive Coding (CPCA)


    CPC-A takes the output of the rnn at timestep t and uses it
    to predict whether a visual embedding (up to) steps in
    the future is the result of taking actions
    {a_t, ..., a_{t+k-1}}.

    This loss has been shown to be useful for learning visual
    representations in embodied AI.
    """

    def __init__(
        self,
        action_space: gym.spaces.Box,
        net: Net,
        k: int = 20,
        time_subsample: int = 6,
        future_subsample: int = 2,
        num_negatives: int = 20,
        loss_scale: float = 0.1,
    ):
        assert (
            not net.is_blind
        ), "CPCA only works for networks with a visual encoder"
        hidden_size = net.output_size
        input_size = net.perception_embedding_size

        super().__init__(
            action_space, hidden_size, k, time_subsample, future_subsample
        )

        self._predictor_first_layers = nn.ModuleList(
            [
                nn.Linear(hidden_size, hidden_size, bias=True),
                nn.Linear(input_size, hidden_size, bias=False),
            ]
        )
        self._predictor = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, 1),
        )

        self.num_negatives = num_negatives
        self.loss_scale = loss_scale

        self.layer_init()

    def forward(self, aux_loss_state, batch):
        (
            future_preds,
            _,
            target_inds,
            future_inds,
            action_valids,
            target_valids,
        ) = super().forward(aux_loss_state, batch)
        device = future_preds.device

        targets = aux_loss_state["perception_embed"]

        future_preds = self._predictor_first_layers[0](
            future_preds.flatten(0, 1)[future_inds]
        )
        positive_inds = target_inds.flatten()[future_inds]
        action_valids = action_valids.flatten()[future_inds]
        target_valids = target_valids.flatten()[future_inds]

        positive_targets = masked_index_select(
            targets, positive_inds, target_valids
        )
        positive_logits = self._predictor(
            future_preds + self._predictor_first_layers[1](positive_targets)
        )

        positive_loss = masked_mean(
            F.binary_cross_entropy_with_logits(
                positive_logits,
                positive_logits.new_full((), 1.0).expand_as(positive_logits),
                reduction="none",
            ),
            target_valids.view(-1, 1),
        )

        negative_inds_probs = targets.new_full(
            (positive_inds.size(0), targets.size(0)), 1.0
        )
        negative_inds_probs[
            torch.arange(
                positive_inds.size(0), device=device, dtype=torch.int64
            ),
            positive_inds,
        ] = 0.0
        negative_inds_probs = negative_inds_probs / negative_inds_probs.sum(
            -1, keepdim=True
        )

        negatives_inds = torch.multinomial(
            negative_inds_probs,
            num_samples=self.num_negatives,
            replacement=self.num_negatives > negative_inds_probs.size(-1),
        )

        negative_targets = _unflatten(
            targets.index_select(0, negatives_inds.flatten()),
            0,
            negatives_inds.size(),
        )
        negative_logits = self._predictor(
            future_preds.unsqueeze(1)
            + self._predictor_first_layers[1](negative_targets)
        )

        negative_loss = F.binary_cross_entropy_with_logits(
            negative_logits,
            negative_logits.new_zeros(()).expand_as(negative_logits),
            reduction="none",
        )
        negative_loss = masked_mean(
            negative_loss, target_valids.view(-1, 1, 1)
        )

        loss = self.loss_scale * (positive_loss + negative_loss)

        return dict(loss=loss)
