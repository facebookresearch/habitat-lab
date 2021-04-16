#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import yaml

from habitat.config import get_config
from habitat.config.default import _C

CFG_TEST = "configs/test/habitat_all_sensors_test.yaml"
CFG_EQA = "configs/test/habitat_mp3d_eqa_test.yaml"
CFG_NEW_KEYS = "configs/test/new_keys_test.yaml"
MULITASK_TEST_FILENAME = "configs/test/habitat_multitask_example.yaml"
MAX_TEST_STEPS_LIMIT = 3


def open_yaml(filename: str):
    with open(filename, "r") as f:
        return yaml.safe_load(f)


def save_yaml(filename: str, obj):
    with open("/tmp/{}".format(filename), "w") as f:
        yaml.dump(obj, f)


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


### Multitask config tests ###
def test_tasks_keep_defaults():
    defaults = _C.TASK.clone()
    cfg = get_config(MULITASK_TEST_FILENAME)
    cfg.defrost()
    cfg.TASKS[0].TYPE = "MyCustomTestTask"
    cfg.freeze()
    assert (
        cfg.TASKS[0].TYPE != cfg.TASK.TYPE
    ), "Each tasks property should be overridable"
    for k in defaults.keys():
        for task in cfg.TASKS:
            assert (
                k in task
            ), "Default property should be inherithed by each task"


def test_global_dataset_config():
    datatype = "MyDatasetType"
    config = open_yaml(MULITASK_TEST_FILENAME)
    for task in config["TASKS"]:
        if "DATASET" in task:
            del task["DATASET"]

    config["DATASET"]["TYPE"] = datatype
    save_yaml("test.yaml", config)
    # load test config
    cfg = get_config("/tmp/test.yaml")
    # make sure each tasks has global dataset config
    for task in cfg.TASKS:
        assert (
            task.DATASET.TYPE == cfg.DATASET.TYPE == datatype
        ), "Each task should inherit global dataset when dataset is not specified"


def test_global_dataset_config_override():
    datatype = "MyDatasetType"
    datapath = "/some/path/"
    config = open_yaml(MULITASK_TEST_FILENAME)
    assert "TASKS" in config
    assert (
        len(config["TASKS"]) > 0
    ), "Need at least one task in tasks to run test"
    for task in config["TASKS"]:
        if "DATASET" in task:
            del task["DATASET"]
    # one tasks needs a different dataset
    config["TASKS"][0]["DATASET"] = {"TYPE": datatype, "DATA_PATH": datapath}
    save_yaml("test.yaml", config)
    # load test config
    cfg = get_config("/tmp/test.yaml")
    # make sure each tasks has global dataset config but the first one
    for i, task in enumerate(cfg.TASKS):
        if i == 0:
            assert (
                task.DATASET.TYPE == datatype != cfg.DATASET.TYPE
            ), "First task should have a different dataset"
        else:
            assert (
                task.DATASET.TYPE == cfg.DATASET.TYPE
            ), "Each task should inherit global dataset when dataset is not specified"
