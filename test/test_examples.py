# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import multiprocessing
import runpy
import sys
from os import path as osp

import pytest


def run_main(*args):
    # patch sys.args
    sys.argv = list(args)
    target = args[0]
    # run_path has one difference with invoking Python from command-line:
    # if the target is a file (rather than a directory), it does not add its
    # parent directory to sys.path. Thus, importing other modules from the
    # same directory is broken unless sys.path is patched here.
    if osp.isfile(target):
        sys.path.insert(0, osp.dirname(target))
    runpy.run_path(target, run_name="__main__")


def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )


def run_main_subproc(args):
    # This test needs to be done in its own process as there is a potentially for
    # an OpenGL context clash otherwise
    mp_ctx = multiprocessing.get_context("spawn")
    proc = mp_ctx.Process(target=run_main, args=args)
    proc.start()
    proc.join()
    assert proc.exitcode == 0


@pytest.mark.skipif(
    not osp.exists(
        "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
    )
    or not osp.exists(
        "data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
    ),
    reason="Requires the habitat-test-scenes",
)
@pytest.mark.parametrize(
    "args",
    [
        ("examples/tutorials/nb_python/Habitat_Lab.py",),
    ],
)
def test_example_modules(args):
    run_main_subproc(args)


@pytest.mark.parametrize(
    "args",
    [
        (
            "examples/interactive_play.py",
            "--no-render",
            "--disable-inverse-kinematics",
        ),
        ("examples/tutorials/nb_python/Habitat2_Quickstart.py",),
        ("examples/tutorials/nb_python/habitat2_gym_tutorial.py",),
    ],
)
def test_rearrange_example_modules(args):
    run_main_subproc(args)


@pytest.mark.skipif(
    not osp.exists(
        "habitat-lab/habitat/config/benchmark/nav/pointnav/pointnav_franka.yaml"
    ),
    reason="Requires the franka point config",
)
@pytest.mark.parametrize(
    "args",
    [
        (
            "examples/franka_example.py",
            "--no-render",
        ),
    ],
)
def test_static_franka_example(args):
    run_main_subproc(args)
