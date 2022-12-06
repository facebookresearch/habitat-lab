#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gc

import pytest

torch = pytest.importorskip("torch")
habitat_baselines = pytest.importorskip("habitat_baselines")

import torch.distributed  # type: ignore[no-redef]

from habitat_baselines.rl.ddppo.ddp_utils import find_free_port, gather_objects


def _worker_fn(
    world_rank: int, world_size: int, port: int, all_same_size: bool
):
    tcp_store = torch.distributed.TCPStore(  # type: ignore
        "127.0.0.1", port, world_size, world_rank == 0
    )
    torch.distributed.init_process_group(
        "gloo", store=tcp_store, rank=world_rank, world_size=world_size
    )
    if all_same_size:
        my_obj = world_rank
    else:
        my_obj = list(range(world_rank + 1))  # type: ignore

    output_rank = max(world_size - 4, 0)

    all_objs = gather_objects(my_obj, output_rank=output_rank)

    if world_rank == output_rank:
        assert all_objs is not None
        assert len(all_objs) == world_size
        for src_rank, obj in enumerate(all_objs):
            if all_same_size:
                assert obj == src_rank
            else:
                assert obj == list(range(src_rank + 1))
    else:
        assert (
            all_objs is None
        ), f"world_rank={world_rank} got not None output with output_rank={output_rank}"

    torch.distributed.barrier()

    torch.distributed.destroy_process_group()
    tcp_store = None
    gc.collect()


@pytest.mark.parametrize("all_same_size", [True, False])
@pytest.mark.parametrize("world_size", [1, 4, 7])
def test_gather_objects(all_same_size: bool, world_size: int):
    torch.multiprocessing.spawn(
        _worker_fn,
        args=(world_size, find_free_port(), all_same_size),
        nprocs=world_size,
    )
