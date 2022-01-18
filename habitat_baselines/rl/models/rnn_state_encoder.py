#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence


def _invert_permutation(permutation: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(permutation)
    output.scatter_(
        0,
        permutation,
        torch.arange(0, permutation.numel(), device=permutation.device),
    )
    return output


def _build_pack_info_from_dones(
    dones: torch.Tensor,
    T: int,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    r"""Create the indexing info needed to make the PackedSequence
    based on the dones.

    PackedSequences are PyTorch's way of supporting a single RNN forward
    call where each input in the batch can have an arbitrary sequence length

    They work as follows: Given the sequences [c], [x, y, z], [a, b],
    we generate data [x, a, c, y, b, z] and batch_sizes [3, 2, 1].  The
    data is a flattened out version of the input sequences (the ordering in
    data is determined by sequence length).  batch_sizes tells you that
    for each index, how many sequences have a length of (index + 1) or greater.

    This method will generate the new index ordering such that you can
    construct the data for a PackedSequence from a (T*N, ...) tensor
    via x.index_select(0, select_inds)
    """
    dones = dones.view(T, -1)
    N = dones.size(1)

    rollout_boundaries = dones.clone().detach()
    # Force a rollout boundary for t=0.  We will use the
    # original dones for masking later, so this is fine
    # and simplifies logic considerably
    rollout_boundaries[0] = True
    rollout_boundaries = rollout_boundaries.nonzero(as_tuple=False)

    # The rollout_boundaries[:, 0]*N will make the episode_starts index into
    # the T*N flattened tensors
    episode_starts = rollout_boundaries[:, 0] * N + rollout_boundaries[:, 1]

    # We need to create a transposed start indexing so we can compute episode lengths
    # As if we make the starts index into a N*T tensor, then starts[1] - starts[0]
    # will compute the length of the 0th episode
    episode_starts_transposed = (
        rollout_boundaries[:, 1] * T + rollout_boundaries[:, 0]
    )
    # Need to sort so the above logic is correct
    episode_starts_transposed, sorted_indices = torch.sort(
        episode_starts_transposed, descending=False
    )

    # Calculate length of episode rollouts
    rollout_lengths = (
        episode_starts_transposed[1:] - episode_starts_transposed[:-1]
    )
    last_len = N * T - episode_starts_transposed[-1]
    rollout_lengths = torch.cat([rollout_lengths, last_len.unsqueeze(0)])
    # Undo the sort above
    rollout_lengths = rollout_lengths.index_select(
        0, _invert_permutation(sorted_indices)
    )

    # Resort in descending order of episode length
    lengths, sorted_indices = torch.sort(rollout_lengths, descending=True)

    # We will want these on the CPU for torch.unique_consecutive,
    # so move now.
    cpu_lengths = lengths.to(device="cpu", non_blocking=True)

    episode_starts = episode_starts.index_select(0, sorted_indices)
    select_inds = torch.empty((T * N), device=dones.device, dtype=torch.int64)

    max_length = int(cpu_lengths[0].item())
    # batch_sizes is *always* on the CPU
    batch_sizes = torch.empty((max_length,), device="cpu", dtype=torch.long)

    offset = 0
    prev_len = 0
    num_valid_for_length = lengths.size(0)

    unique_lengths = torch.unique_consecutive(cpu_lengths)
    # Iterate over all unique lengths in reverse as they sorted
    # in decreasing order
    for next_len in reversed(unique_lengths):
        valids = lengths[0:num_valid_for_length] > prev_len
        num_valid_for_length = int(valids.float().sum())

        batch_sizes[prev_len:next_len] = num_valid_for_length

        # Creates this array
        # [step * N + start for step in range(prev_len, next_len)
        #                   for start in episode_starts[0:num_valid_for_length]
        # * N because each step is separated by N elements
        new_inds = (
            torch.arange(
                prev_len, next_len, device=episode_starts.device
            ).view(next_len - prev_len, 1)
            * N
            + episode_starts[0:num_valid_for_length].view(
                1, num_valid_for_length
            )
        ).view(-1)

        select_inds[offset : offset + new_inds.numel()] = new_inds

        offset += new_inds.numel()

        prev_len = next_len

    # Make sure we have an index for all elements
    assert offset == T * N

    # This is used in conjunction with episode_starts to get
    # the RNN hidden states
    rnn_state_batch_inds = episode_starts % N
    # This indicates that a given episode is the last one
    # in that rollout.  In other words, there are N places
    # where this is True, and for each n, True indicates
    # that this episode is the last contiguous block of experience,
    # This is needed for getting the correct hidden states after
    # the RNN forward pass
    last_episode_in_batch_mask = (
        (episode_starts + (lengths - 1) * N) // N
    ) == (T - 1)

    return (
        select_inds,
        batch_sizes,
        episode_starts,
        rnn_state_batch_inds,
        last_episode_in_batch_mask,
    )


def build_rnn_inputs(
    x: torch.Tensor, not_dones: torch.Tensor, rnn_states: torch.Tensor
) -> Tuple[
    PackedSequence, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    r"""Create a PackedSequence input for an RNN such that each
    set of steps that are part of the same episode are all part of
    a batch in the PackedSequence.

    Use the returned select_inds and build_rnn_out_from_seq to invert this.

    :param x: A (T * N, -1) tensor of the data to build the PackedSequence out of
    :param not_dones: A (T * N) tensor where not_dones[i] == False indicates an episode is done
    :param rnn_states: A (-1, N, -1) tensor of the rnn_hidden_states

    :return: tuple(x_seq, rnn_states, select_inds, rnn_state_batch_inds, last_episode_in_batch_mask)
        WHERE
        x_seq is the PackedSequence version of x to pass to the RNN

        rnn_states are the corresponding rnn state

        select_inds can be passed to build_rnn_out_from_seq to retrieve the
            RNN output

        rnn_state_batch_inds indicates which of the rollouts in the batch a hidden
            state came from/is for

        last_episode_in_batch_mask indicates if an episode is the last in that batch.
            There will be exactly N places where this is True

    """

    N = rnn_states.size(1)
    T = x.size(0) // N
    dones = torch.logical_not(not_dones)

    (
        select_inds,
        batch_sizes,
        episode_starts,
        rnn_state_batch_inds,
        last_episode_in_batch_mask,
    ) = _build_pack_info_from_dones(dones.detach().to(device="cpu"), T)

    select_inds = select_inds.to(device=x.device)
    episode_starts = episode_starts.to(device=x.device)
    rnn_state_batch_inds = rnn_state_batch_inds.to(device=x.device)
    last_episode_in_batch_mask = last_episode_in_batch_mask.to(device=x.device)

    x_seq = PackedSequence(
        x.index_select(0, select_inds), batch_sizes, None, None
    )

    # Just select the rnn_states by batch index, the masking bellow will set things
    # to zero in the correct locations
    rnn_states = rnn_states.index_select(1, rnn_state_batch_inds)
    # Now zero things out in the correct locations
    rnn_states = torch.where(
        not_dones.view(1, -1, 1).index_select(1, episode_starts),
        rnn_states,
        rnn_states.new_zeros(()),
    )

    return (
        x_seq,
        rnn_states,
        select_inds,
        rnn_state_batch_inds,
        last_episode_in_batch_mask,
    )


def build_rnn_out_from_seq(
    x_seq: PackedSequence,
    hidden_states,
    select_inds,
    rnn_state_batch_inds,
    last_episode_in_batch_mask,
    N: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Construct the output of the rnn from a packed sequence returned by
        forward propping an RNN on the packed sequence returned by :ref:`build_rnn_inputs`.

    :param x_seq: The packed sequence output from the rnn
    :param hidden_statess: The hidden states output from the rnn
    :param select_inds: Returned from :ref:`build_rnn_inputs`
    :param rnn_state_batch_inds: Returned from :ref:`build_rnn_inputs`
    :param last_episode_in_batch_mask: Returned from :ref:`build_rnn_inputs`
    :param N: The number of simulator instances in the batch of experience.
    """
    x = x_seq.data.index_select(0, _invert_permutation(select_inds))

    last_hidden_states = torch.masked_select(
        hidden_states,
        last_episode_in_batch_mask.view(1, hidden_states.size(1), 1),
    ).view(hidden_states.size(0), N, hidden_states.size(2))
    output_hidden_states = torch.empty_like(last_hidden_states)
    scatter_inds = (
        torch.masked_select(rnn_state_batch_inds, last_episode_in_batch_mask)
        .view(1, N, 1)
        .expand_as(output_hidden_states)
    )
    output_hidden_states.scatter_(1, scatter_inds, last_hidden_states)

    return x, output_hidden_states


class RNNStateEncoder(nn.Module):
    r"""RNN encoder for use with RL and possibly IL.

    The main functionality this provides over just using PyTorch's RNN interface directly
    is that it takes an addition masks input that resets the hidden state between two adjacent
    timesteps to handle episodes ending in the middle of a rollout.
    """
    rnn: torch.nn.Module

    def layer_init(self):
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def pack_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def unpack_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def single_forward(
        self, x, hidden_states, masks
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward for a non-sequence input"""

        hidden_states = torch.where(
            masks.view(1, -1, 1), hidden_states, hidden_states.new_zeros(())
        )

        x, hidden_states = self.rnn(
            x.unsqueeze(0), self.unpack_hidden(hidden_states)
        )
        hidden_states = self.pack_hidden(hidden_states)

        x = x.squeeze(0)
        return x, hidden_states

    def seq_forward(
        self, x, hidden_states, masks
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward for a sequence of length T

        Args:
            x: (T, N, -1) Tensor that has been flattened to (T * N, -1)
            hidden_states: The starting hidden state.
            masks: The masks to be applied to hidden state at every timestep.
                A (T, N) tensor flatten to (T * N)
        """
        N = hidden_states.size(1)

        (
            x_seq,
            hidden_states,
            select_inds,
            rnn_state_batch_inds,
            last_episode_in_batch_mask,
        ) = build_rnn_inputs(x, masks, hidden_states)

        x_seq, hidden_states = self.rnn(
            x_seq, self.unpack_hidden(hidden_states)
        )
        hidden_states = self.pack_hidden(hidden_states)

        x, hidden_states = build_rnn_out_from_seq(
            x_seq,
            hidden_states,
            select_inds,
            rnn_state_batch_inds,
            last_episode_in_batch_mask,
            N,
        )

        return x, hidden_states

    def forward(
        self, x, hidden_states, masks
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.permute(1, 0, 2)
        if x.size(0) == hidden_states.size(1):
            x, hidden_states = self.single_forward(x, hidden_states, masks)
        else:
            x, hidden_states = self.seq_forward(x, hidden_states, masks)

        hidden_states = hidden_states.permute(1, 0, 2)

        return x, hidden_states


class LSTMStateEncoder(RNNStateEncoder):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
    ):
        super().__init__()

        self.num_recurrent_layers = num_layers * 2

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        self.layer_init()

    def pack_hidden(
        self, hidden_states: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        return torch.cat(hidden_states, 0)

    def unpack_hidden(
        self, hidden_states
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lstm_states = torch.chunk(hidden_states, 2, 0)
        return (lstm_states[0], lstm_states[1])


class GRUStateEncoder(RNNStateEncoder):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
    ):
        super().__init__()

        self.num_recurrent_layers = num_layers

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        self.layer_init()


def build_rnn_state_encoder(
    input_size: int,
    hidden_size: int,
    rnn_type: str = "GRU",
    num_layers: int = 1,
):
    r"""Factory for :ref:`RNNStateEncoder`.  Returns one with either a GRU or LSTM based on
        the specified RNN type.

    :param input_size: The input size of the RNN
    :param hidden_size: The hidden dimension of the RNN
    :param rnn_types: The type of the RNN cell.  Can either be GRU or LSTM
    :param num_layers: The number of RNN layers.
    """
    rnn_type = rnn_type.lower()
    if rnn_type == "gru":
        return GRUStateEncoder(input_size, hidden_size, num_layers)
    elif rnn_type == "lstm":
        return LSTMStateEncoder(input_size, hidden_size, num_layers)
    else:
        raise RuntimeError(f"Did not recognize rnn type '{rnn_type}'")
