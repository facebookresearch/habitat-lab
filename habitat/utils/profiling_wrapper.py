#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""Wrappers for habitat_sim profiling_utils functions. The wrappers are no-ops
if habitat_sim isn't installed.

Example of capturing an Nsight Systems profile with Habitat-lab:
export HABITAT_PROFILING=1
export NSYS_NVTX_PROFILER_REGISTER_ONLY=0  # required when using capture range
path/to/nvidia/nsight-systems/bin/nsys profile --sample=none --trace=nvtx --trace-fork-before-exec=true --capture-range=nvtx -p "habitat_capture_range" --stop-on-range-end=true --output=my_profile --export=sqlite python habitat_baselines/run.py --exp-config habitat_baselines/config/pointnav/ppo_pointnav.yaml --run-type train PROFILING.CAPTURE_START_STEP 200 PROFILING.NUM_STEPS_TO_CAPTURE 100
# look for my_profile.qdrep in working directory
"""

from contextlib import ContextDecorator

try:
    from habitat_sim.utils import profiling_utils
except ImportError:
    profiling_utils = None


def configure(capture_start_step=-1, num_steps_to_capture=-1):
    r"""Wrapper for habitat_sim profiling_utils.configure"""
    if profiling_utils:
        profiling_utils.configure(capture_start_step, num_steps_to_capture)


def on_start_step():
    r"""Wrapper for habitat_sim profiling_utils.on_start_step"""
    if profiling_utils:
        profiling_utils.on_start_step()


def range_push(msg: str):
    r"""Wrapper for habitat_sim profiling_utils.range_push"""
    if profiling_utils:
        profiling_utils.range_push(msg)


def range_pop():
    r"""Wrapper for habitat_sim profiling_utils.range_pop"""
    if profiling_utils:
        profiling_utils.range_pop()


class RangeContext(ContextDecorator):
    r"""Annotate a range for profiling. Use as a function decorator or in a with
    statement. See habitat_sim profiling_utils.
    """

    def __init__(self, msg: str):
        self._msg = msg

    def __enter__(self):
        range_push(self._msg)
        return self

    def __exit__(self, *exc):
        range_pop()
        return False
