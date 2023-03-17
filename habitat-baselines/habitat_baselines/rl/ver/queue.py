#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

try:
    import faster_fifo

    use_faster_fifo = True
except ImportError:
    use_faster_fifo = False


if not TYPE_CHECKING and use_faster_fifo:
    # This package contains the implementation
    # for pickling FasterFifo with ForkingPickler,
    # which allows the queues to be passed to processes.
    import faster_fifo_reduction  # noqa: F401

    BatchedQueue = faster_fifo.Queue
else:
    import queue
    import time
    import warnings

    import torch

    warnings.warn(
        "Unable to import faster_fifo."
        " Using the fallback. This may reduce performance."
    )

    class BatchedQueue(torch.multiprocessing.Queue):
        def get_many(
            self,
            block=True,
            timeout=10.0,
            max_messages_to_get=1_000_000_000,
        ):
            msgs = [self.get(block, timeout)]
            while len(msgs) < max_messages_to_get:
                try:
                    msgs.append(self.get_nowait())
                except queue.Empty:
                    break

            return msgs

        def put_many(self, xs, block=True, timeout=10.0):
            t_start = time.perf_counter()
            n_put = 0
            for x in xs:
                self.put(x, block, timeout - (t_start - time.perf_counter()))
                n_put += 1

            if n_put != len(xs):
                raise RuntimeError(
                    f"Couldn't put all. Put {n_put}, needed to put {len(xs)}"
                )
