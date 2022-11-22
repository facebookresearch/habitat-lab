#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence

from habitat_baselines.common.tensor_dict import TensorDict


def _invert_permutation(permutation: torch.Tensor) -> torch.Tensor:
    orig_size = permutation.size()
    permutation = permutation.view(-1)
    output = torch.empty_like(permutation)
    output.scatter_(
        0,
        permutation,
        torch.arange(0, permutation.numel(), device=permutation.device),
    )
    return output.view(orig_size)


def _np_invert_permutation(permutation: np.ndarray) -> np.ndarray:
    return np.argsort(permutation.ravel()).reshape(permutation.shape)


# This is some pretty wild code. I recommend you just trust
# the unit test on it and leave it be.
def build_pack_info_from_episode_ids(
    episode_ids: np.ndarray,
    environment_ids: np.ndarray,
    step_ids: np.ndarray,
) -> Dict[str, np.ndarray]:
    r"""Create the indexing info needed to make the PackedSequence
    based on the dones.

    PackedSequences are PyTorch's way of supporting a single RNN forward
    call where each input in the batch can have an arbitrary sequence length

    They work as follows: Given the sequences [c], [x, y, z], [a, b],
    we generate data [x, a, c, y, b, z] and num_seqs_at_step [3, 2, 1].  The
    data is a flattened out version of the input sequences (the ordering in
    data is determined by sequence length).  num_seqs_at_step tells you that
    for each index, how many sequences have a length of (index + 1) or greater.

    This method will generate the new index ordering such that you can
    construct the data for a PackedSequence from a (T*N, ...) tensor
    via x.index_select(0, select_inds)
    """
    # make episode_ids globally unique. This will make things easier
    episode_ids = episode_ids * (environment_ids.max() + 1) + environment_ids
    unsorted_episode_ids = episode_ids
    # Sort in increasing order of (episode ID, step ID).  This will
    # put things into an order such that each episode is a contiguous
    # block. This makes all the following logic MUCH easier
    sort_keys = episode_ids * (step_ids.max() + 1) + step_ids
    assert np.unique(sort_keys).size == sort_keys.size
    episode_id_sorting = np.argsort(
        episode_ids * (step_ids.max() + 1) + step_ids
    )
    episode_ids = episode_ids[episode_id_sorting]

    unique_episode_ids, sequence_lengths = np.unique(
        episode_ids, return_counts=True
    )
    # Exclusive cumsum
    sequence_starts = np.cumsum(sequence_lengths) - sequence_lengths

    sorted_indices = np.argsort(-sequence_lengths)
    lengths = sequence_lengths[sorted_indices]
    #  print(lengths)

    unique_episode_ids = unique_episode_ids[sorted_indices]
    sequence_starts = sequence_starts[sorted_indices]

    max_length = int(lengths[0])

    select_inds = np.empty((episode_ids.size,), dtype=np.int64)

    # num_seqs_at_step is *always* on the CPU
    num_seqs_at_step = np.empty((max_length,), dtype=np.int64)

    offset = 0
    prev_len = 0
    num_valid_for_length = lengths.shape[0]
    #  print(lengths)

    for next_len in np.unique(lengths):
        num_valid_for_length = np.count_nonzero(
            lengths[0:num_valid_for_length] > prev_len
        )

        num_seqs_at_step[prev_len:next_len] = num_valid_for_length

        new_inds = (
            sequence_starts[0:num_valid_for_length][np.newaxis, :]
            + np.arange(prev_len, next_len)[:, np.newaxis]
        ).reshape(-1)

        select_inds[offset : offset + new_inds.size] = new_inds

        offset += new_inds.size

        prev_len = int(next_len)

    assert offset == select_inds.size

    select_inds = episode_id_sorting[select_inds]
    sequence_starts = select_inds[0 : num_seqs_at_step[0]]

    episode_environment_ids = environment_ids[sequence_starts]
    unique_environment_ids, rnn_state_batch_inds = np.unique(
        episode_environment_ids, return_inverse=True
    )
    episode_ids_for_starts = unsorted_episode_ids[sequence_starts]
    last_sequence_in_batch_mask = np.zeros_like(episode_environment_ids == 0)
    first_sequence_in_batch_mask = np.zeros_like(last_sequence_in_batch_mask)
    first_step_for_env = []
    for env_id in unique_environment_ids:
        env_eps = episode_environment_ids == env_id
        env_eps_ids = episode_ids_for_starts[env_eps]

        last_sequence_in_batch_mask[env_eps] = env_eps_ids == env_eps_ids.max()
        first_ep_mask = env_eps_ids == env_eps_ids.min()
        first_sequence_in_batch_mask[env_eps] = first_ep_mask

        first_step_for_env.append(
            sequence_starts[env_eps][first_ep_mask].item()
        )

    return {
        "select_inds": select_inds,
        "num_seqs_at_step": num_seqs_at_step,
        "sequence_starts": sequence_starts,
        "sequence_lengths": lengths,
        "rnn_state_batch_inds": rnn_state_batch_inds,
        "last_sequence_in_batch_mask": last_sequence_in_batch_mask,
        "first_sequence_in_batch_mask": first_sequence_in_batch_mask,
        "last_sequence_in_batch_inds": np.nonzero(last_sequence_in_batch_mask)[
            0
        ],
        "first_episode_in_batch_inds": np.nonzero(
            first_sequence_in_batch_mask
        )[0],
        "first_step_for_env": np.asarray(first_step_for_env),
    }


def build_pack_info_from_dones(dones: np.ndarray) -> Dict[str, np.ndarray]:
    T, N = dones.shape
    episode_ids = np.cumsum(dones, 0)
    environment_ids = np.arange(N).reshape(1, N).repeat(T, 0)
    # Technically the step_ids should reset to 0 after each done,
    # but build_pack_info_from_episode_ids doesn't depend on this
    # so we don't do it.
    step_ids = np.arange(T).reshape(T, 1).repeat(N, 1)

    return build_pack_info_from_episode_ids(
        episode_ids.reshape(-1),
        environment_ids.reshape(-1),
        step_ids.reshape(-1),
    )


def build_rnn_build_seq_info(
    device: torch.device, build_fn_result: Dict[str, np.ndarray]
) -> TensorDict:
    r"""Creates the dict with the build pack seq results."""
    rnn_build_seq_info = TensorDict()
    for k, v_n in build_fn_result.items():
        v = torch.from_numpy(v_n)
        # We keep the CPU side
        # tensor as well. This makes various things
        # easier and some things need to be on the CPU
        rnn_build_seq_info[f"cpu_{k}"] = v
        rnn_build_seq_info[k] = v.to(device=device)

    return rnn_build_seq_info


def build_rnn_inputs(
    x: torch.Tensor,
    rnn_states: torch.Tensor,
    not_dones,
    rnn_build_seq_info,
) -> Tuple[PackedSequence, torch.Tensor,]:
    r"""Create a PackedSequence input for an RNN such that each
    set of steps that are part of the same episode are all part of
    a batch in the PackedSequence.

    Use the returned select_inds and build_rnn_out_from_seq to invert this.

    :param x: A (T * N, -1) tensor of the data to build the PackedSequence out of
    :param rnn_states: A (-1, N, -1) tensor of the rnn_hidden_states

    :return: tuple(x_seq, rnn_states, select_inds, rnn_state_batch_inds, last_sequence_in_batch_mask)
        WHERE
        x_seq is the PackedSequence version of x to pass to the RNN

        rnn_states are the corresponding rnn state

        select_inds can be passed to build_rnn_out_from_seq to retrieve the
            RNN output

        rnn_state_batch_inds indicates which of the rollouts in the batch a hidden
            state came from/is for

        last_sequence_in_batch_mask indicates if an episode is the last in that batch.
            There will be exactly N places where this is True

    """

    select_inds = rnn_build_seq_info["select_inds"]
    num_seqs_at_step = rnn_build_seq_info["cpu_num_seqs_at_step"]

    x_seq = PackedSequence(
        x.index_select(0, select_inds), num_seqs_at_step, None, None
    )

    rnn_state_batch_inds = rnn_build_seq_info["rnn_state_batch_inds"]
    sequence_starts = rnn_build_seq_info["sequence_starts"]

    # Just select the rnn_states by batch index, the masking bellow will set things
    # to zero in the correct locations
    rnn_states = rnn_states.index_select(1, rnn_state_batch_inds)
    # Now zero things out in the correct locations
    rnn_states.masked_fill_(
        torch.logical_not(
            not_dones.view(1, -1, 1).index_select(1, sequence_starts)
        ),
        0,
    )

    return (
        x_seq,
        rnn_states,
    )


def build_rnn_out_from_seq(
    x_seq: PackedSequence,
    hidden_states,
    rnn_build_seq_info,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Construct the output of the rnn from a packed sequence returned by
        forward propping an RNN on the packed sequence returned by :ref:`build_rnn_inputs`.

    :param x_seq: The packed sequence output from the rnn
    :param hidden_statess: The hidden states output from the rnn
    :param select_inds: Returned from :ref:`build_rnn_inputs`
    :param rnn_state_batch_inds: Returned from :ref:`build_rnn_inputs`
    :param last_sequence_in_batch_mask: Returned from :ref:`build_rnn_inputs`
    :param N: The number of simulator instances in the batch of experience.
    """
    select_inds = rnn_build_seq_info["select_inds"]
    x = x_seq.data.index_select(0, _invert_permutation(select_inds))

    last_sequence_in_batch_inds = rnn_build_seq_info[
        "last_sequence_in_batch_inds"
    ]
    rnn_state_batch_inds = rnn_build_seq_info["rnn_state_batch_inds"]
    output_hidden_states = hidden_states.index_select(
        1,
        last_sequence_in_batch_inds[
            _invert_permutation(
                rnn_state_batch_inds[last_sequence_in_batch_inds]
            )
        ],
    )

    return x, output_hidden_states


class RNNStateEncoder(nn.Module):
    r"""RNN encoder for use with RL and possibly IL.

    The main functionality this provides over just using PyTorch's RNN interface directly
    is that it takes an addition masks input that resets the hidden state between two adjacent
    timesteps to handle episodes ending in the middle of a rollout.
    """

    def layer_init(self):
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def pack_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def unpack_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.contiguous()

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
        self,
        x,
        hidden_states,
        masks,
        rnn_build_seq_info: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward for a sequence of length T

        Args:
            x: (T, N, -1) Tensor that has been flattened to (T * N, -1)
            hidden_states: The starting hidden state.
            masks: The masks to be applied to hidden state at every timestep.
                A (T, N) tensor flatten to (T * N)
        """

        (
            x_seq,
            hidden_states,
        ) = build_rnn_inputs(x, hidden_states, masks, rnn_build_seq_info)

        rnn_ret = self.rnn(x_seq, self.unpack_hidden(hidden_states))
        x_seq: PackedSequence = rnn_ret[0]
        hidden_states: torch.Tensor = rnn_ret[1]
        hidden_states = self.pack_hidden(hidden_states)

        x, hidden_states = build_rnn_out_from_seq(
            x_seq,
            hidden_states,
            rnn_build_seq_info,
        )

        return x, hidden_states

    def forward(
        self,
        x,
        hidden_states,
        masks,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.permute(1, 0, 2)
        if x.size(0) == hidden_states.size(1):
            assert rnn_build_seq_info is None
            x, hidden_states = self.single_forward(x, hidden_states, masks)
        else:
            assert rnn_build_seq_info is not None
            x, hidden_states = self.seq_forward(
                x, hidden_states, masks, rnn_build_seq_info
            )

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
        lstm_states = torch.chunk(hidden_states.contiguous(), 2, 0)
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
