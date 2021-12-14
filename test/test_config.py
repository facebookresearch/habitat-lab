#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from glob import glob

import pytest

import habitat
from habitat.config.default import get_config

CFG_TEST = "configs/test/habitat_all_sensors_test.yaml"
CFG_EQA = "configs/test/habitat_mp3d_eqa_test.yaml"
CFG_NEW_KEYS = "configs/test/new_keys_test.yaml"
MAX_TEST_STEPS_LIMIT = 3


def test_merged_configs():
    test_config = get_config(CFG_TEST)
    eqa_config = get_config(CFG_EQA)
    merged_config = get_config("{},{}".format(CFG_TEST, CFG_EQA))
    assert merged_config.TASK.TYPE == eqa_config.TASK.TYPE
    assert (
        merged_config.ENVIRONMENT.MAX_EPISODE_STEPS
        == test_config.ENVIRONMENT.MAX_EPISODE_STEPS
    )


def test_new_keys_merged_configs():
    test_config = get_config(CFG_TEST)
    new_keys_config = get_config(CFG_NEW_KEYS)
    merged_config = get_config("{},{}".format(CFG_TEST, CFG_NEW_KEYS))
    assert (
        merged_config.TASK.MY_NEW_TASK_PARAM
        == new_keys_config.TASK.MY_NEW_TASK_PARAM
    )
    assert (
        merged_config.ENVIRONMENT.MAX_EPISODE_STEPS
        == test_config.ENVIRONMENT.MAX_EPISODE_STEPS
    )


def test_overwrite_options():
    for steps_limit in range(MAX_TEST_STEPS_LIMIT):
        config = get_config(
            config_paths=CFG_TEST,
            opts=["ENVIRONMENT.MAX_EPISODE_STEPS", steps_limit],
        )
        assert (
            config.ENVIRONMENT.MAX_EPISODE_STEPS == steps_limit
        ), "Overwriting of config options failed."


@pytest.mark.parametrize(
    "config_path",
    glob("configs/**/*.yaml", recursive=True),
)
def test_no_core_config_has_non_default_keys(config_path):
    configs_allowed_to_have_non_default_keys = [
        # new_keys_test.yaml excluded since it explicitely uses
        # keys not present in the default for testing purposes
        "configs/test/new_keys_test.yaml",
        # Planning Domain Definition Language configs are
        # excluded since they do not implement the default config
        "configs/tasks/rearrange/pddl/replica_cad_domain.yaml",
        "configs/tasks/rearrange/pddl/nav_pick.yaml",
        "configs/tasks/rearrange/pddl/tidy_house.yaml",
        # Trainer excluded because does not use the default config
        "configs/baselines/ppo.yaml",
    ]
    if config_path in configs_allowed_to_have_non_default_keys:
        return
    from habitat.config.default import _C

    # We manually disallow new keys when merging to make sure all keys
    # are in the default config
    _C.set_new_allowed(False)
    try:
        habitat.get_config(config_path)
    finally:
        _C.set_new_allowed(True)
