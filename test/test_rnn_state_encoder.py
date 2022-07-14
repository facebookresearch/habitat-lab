#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest

try:
    import torch
except ImportError:
    torch = None


@pytest.mark.skipif(torch is None, reason="Test requires pytorch")
def test_rnn_state_encoder():
    from habitat_baselines.rl.models.rnn_state_encoder import (
        build_pack_info_from_dones,
        build_rnn_build_seq_info,
        build_rnn_state_encoder,
    )

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    rnn_state_encoder = build_rnn_state_encoder(32, 32, num_layers=2).to(
        device=device
    )
    rnn = rnn_state_encoder.rnn
    with torch.no_grad():
        for T in [1, 2, 4, 8, 16, 32, 64, 3, 13, 31]:
            for N in [1, 2, 4, 8, 3, 5]:
                masks = torch.rand((T, N, 1), device=device) > (1.0 / 25.0)
                if T == 1:
                    rnn_build_seq_info = None
                else:
                    rnn_build_seq_info = build_rnn_build_seq_info(
                        device,
                        build_fn_result=build_pack_info_from_dones(
                            torch.logical_not(masks).view(T, N).cpu().numpy()
                        ),
                    )

                inputs = torch.randn((T, N, 32), device=device)
                hidden_states = torch.randn(
                    rnn_state_encoder.num_recurrent_layers,
                    N,
                    32,
                    device=device,
                )

                outputs, out_hiddens = rnn_state_encoder(
                    inputs.flatten(0, 1),
                    hidden_states.permute(1, 0, 2),
                    masks.flatten(0, 1),
                    rnn_build_seq_info,
                )
                out_hiddens = out_hiddens.permute(1, 0, 2)

                reference_ouputs = []
                reference_hiddens = hidden_states.clone()
                for t in range(T):
                    reference_hiddens = torch.where(
                        masks[t].view(1, -1, 1),
                        reference_hiddens,
                        reference_hiddens.new_zeros(()),
                    )

                    x, reference_hiddens = rnn(
                        inputs[t : t + 1], reference_hiddens
                    )

                    reference_ouputs.append(x.squeeze(0))

                reference_ouputs = torch.stack(reference_ouputs, 0).flatten(
                    0, 1
                )

                assert (
                    torch.norm(reference_ouputs - outputs).item() < 1e-3
                ), "Failed on (T={}, N={})".format(T, N)
                assert (
                    torch.norm(reference_hiddens - out_hiddens).item() < 1e-3
                ), "Failed on (T={}, N={})".format(T, N)
