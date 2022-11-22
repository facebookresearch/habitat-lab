#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from glob import glob

import pytest

import habitat
from habitat.config.default import _C, _HABITAT_CFG_DIR, get_config

CFG_TEST = "test/habitat_all_sensors_test.yaml"
CFG_EQA = "test/habitat_mp3d_eqa_test.yaml"
MAX_TEST_STEPS_LIMIT = 3


def test_overwrite_options():
    for steps_limit in range(MAX_TEST_STEPS_LIMIT):
        config = get_config(
            config_paths=CFG_TEST,
            overrides=[f"habitat.environment.max_episode_steps={steps_limit}"],
        )
        assert (
            config.habitat.environment.max_episode_steps == steps_limit
        ), "Overwriting of config options failed."


CONFIGS_ALLOWED_TO_HAVE_NON_DEFAULT_KEYS = [
    # Trainer excluded because does not use the default config
    _HABITAT_CFG_DIR + "/baselines/ppo.yaml",
    _HABITAT_CFG_DIR + "/task/rearrange/rearrange_easy_multi_agent.yaml",
    # Planning Domain Definition Language configs are
    # excluded since they do not implement the default config
] + glob(_HABITAT_CFG_DIR + "/**/pddl/*.yaml", recursive=True)


@pytest.mark.parametrize(
    "config_path",
    glob(_HABITAT_CFG_DIR + "/benchmark/**/*.yaml", recursive=True),
)
def test_no_core_config_has_non_default_keys(config_path):
    if config_path in CONFIGS_ALLOWED_TO_HAVE_NON_DEFAULT_KEYS:
        pytest.skip(f"File {config_path} manually excluded from test")
    # We manually disallow new keys when merging to make sure all keys
    # are in the default config
    _C.set_new_allowed(False)
    try:
        habitat.get_config(config_path)
    except KeyError as e:
        raise KeyError(f"Failed for config {config_path}") from e
    finally:
        _C.set_new_allowed(True)
