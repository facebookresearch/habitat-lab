#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from glob import glob

import pytest

try:
    from habitat_baselines.run import run_exp
    from habitat_baselines.common.base_trainer import BaseRLTrainer
    from habitat_baselines.config.default import get_config

    baseline_installed = True
except ImportError:
    baseline_installed = False


@pytest.mark.skipif(
    not baseline_installed, reason="baseline sub-module not installed"
)
@pytest.mark.parametrize(
    "test_cfg_path,mode,gpu2gpu",
    itertools.product(
        glob("habitat_baselines/config/test/*"),
        ["train", "eval"],
        [True, False],
    ),
)
def test_trainers(test_cfg_path, mode, gpu2gpu):
    if gpu2gpu:
        try:
            import habitat_sim
        except ImportError:
            pytest.skip("GPU-GPU requires Habitat-Sim")

        if not habitat_sim.cuda_enabled:
            pytest.skip("GPU-GPU requires CUDA")

    run_exp(
        test_cfg_path,
        mode,
        ["TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.GPU_GPU", str(gpu2gpu)],
    )


@pytest.mark.skipif(
    not baseline_installed, reason="baseline sub-module not installed"
)
def test_eval_config():
    ckpt_opts = ["VIDEO_OPTION", "[]"]
    eval_opts = ["VIDEO_OPTION", "['disk']"]

    ckpt_cfg = get_config(None, ckpt_opts)
    assert ckpt_cfg.VIDEO_OPTION == []
    assert ckpt_cfg.CMD_TRAILING_OPTS == ["VIDEO_OPTION", "[]"]

    eval_cfg = get_config(None, eval_opts)
    assert eval_cfg.VIDEO_OPTION == ["disk"]
    assert eval_cfg.CMD_TRAILING_OPTS == ["VIDEO_OPTION", "['disk']"]

    trainer = BaseRLTrainer(get_config())
    assert trainer.config.VIDEO_OPTION == ["disk", "tensorboard"]
    returned_config = trainer._setup_eval_config(checkpoint_config=ckpt_cfg)
    assert returned_config.VIDEO_OPTION == []

    trainer = BaseRLTrainer(eval_cfg)
    returned_config = trainer._setup_eval_config(ckpt_cfg)
    assert returned_config.VIDEO_OPTION == ["disk"]
