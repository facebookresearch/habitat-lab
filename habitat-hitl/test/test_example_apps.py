#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing
import runpy
import sys
from os import path as osp

import pytest


def run_main(*args):
    sys.argv = list(args)
    target = args[0]
    if osp.isfile(target):
        sys.path.insert(0, osp.dirname(target))
    runpy.run_path(target, run_name="__main__")


def run_main_as_subprocess(args):
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=run_main, args=args)
    process.start()
    process.join()
    assert process.exitcode == 0


@pytest.mark.skipif(
    not osp.exists("data/scene_datasets/hssd-hab"),
    reason="Requires public Habitat-HSSD scene dataset. TODO: should be updated to a new dataset.",
)
@pytest.mark.parametrize(
    "args",
    [
        (
            "examples/hitl/basic_viewer/basic_viewer.py",
            "--config-dir",
            "habitat-hitl/test/config",
            "+experiment=smoke_test",
        ),
    ],
)
def test_hitl_example_basic_viewer(args):
    run_main_as_subprocess(args)


@pytest.mark.skipif(
    not osp.exists("data/scene_datasets/hssd-hab"),
    reason="Requires public Habitat-HSSD scene dataset. TODO: should be updated to a new dataset.",
)
@pytest.mark.parametrize(
    "args",
    [
        (
            "examples/hitl/minimal/minimal.py",
            "--config-dir",
            "habitat-hitl/test/config",
            "+experiment=smoke_test",
        ),
    ],
)
def test_hitl_example_minimal(args):
    run_main_as_subprocess(args)


@pytest.mark.skipif(
    not osp.exists("data/scene_datasets/hssd-hab"),
    reason="Requires public Habitat-HSSD scene dataset. TODO: should be updated to a new dataset.",
)
@pytest.mark.parametrize(
    "args",
    [
        (
            "examples/hitl/pick_throw_vr/pick_throw_vr.py",
            "--config-dir",
            "habitat-hitl/test/config",
            "+experiment=smoke_test",
        ),
    ],
)
def test_hitl_example_pick_throw_vr(args):
    run_main_as_subprocess(args)


@pytest.mark.skipif(
    not osp.exists("data/scene_datasets/hssd-hab"),
    reason="Requires public Habitat-HSSD scene dataset. TODO: should be updated to a new dataset.",
)
@pytest.mark.parametrize(
    "args",
    [
        (
            "examples/hitl/rearrange/rearrange.py",
            "--config-dir",
            "habitat-hitl/test/config",
            "+experiment=smoke_test",
        ),
    ],
)
def test_hitl_example_rearrange(args):
    run_main_as_subprocess(args)


@pytest.mark.skip(reason="Cannot currently be tested.")
@pytest.mark.skipif(
    not osp.exists("data/scene_datasets/hssd-hab"),
    reason="Requires public Habitat-HSSD scene dataset. TODO: should be updated to a new dataset.",
)
@pytest.mark.parametrize(
    "args",
    [
        (
            "examples/hitl/rearrange_v2/main.py",
            "--config-dir",
            "habitat-hitl/test/config",
            "+experiment=smoke_test",
        ),
    ],
)
def test_hitl_example_rearrange_v2(args):
    run_main_as_subprocess(args)
