#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os.path as osp
from typing import TYPE_CHECKING, Any, Dict, List

from omegaconf import OmegaConf

import habitat
from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.rearrange.rearrange_task import RearrangeTask

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from habitat.datasets.rearrange.rearrange_dataset import RearrangeDatasetV0


TASK_CONFIGS_DIR = "benchmark/rearrange/"
TASK_IGNORE_KEYS = ["task_spec", "task_spec_base_path", "pddl_domain_def"]


def create_task_object(
    task_cls_name: str,
    task_config_path: str,
    cur_config: "DictConfig",
    cur_env: RearrangeTask,
    cur_dataset: "RearrangeDatasetV0",
    should_super_reset: bool,
    task_kwargs: Dict[str, Any],
    episode: Episode,
    task_config_args: Dict[str, Any],
) -> RearrangeTask:
    """
    Creates a task to be used within another task. Used when a task needs to be simulated within another task. For example, this is used to get the starting state of another task as a navigation goal in the Habitat 2.0 navigation task. The loaded task uses the information and dataset from the main task (which is also passed into this function).

    :param task_cls_name: The name of the task to load.
    :param task_config_path: The path to the config for the task to load.
    :param cur_config: The config for the main task.
    :param cur_env: The main task.
    """
    task_cls = registry.get_task(task_cls_name)

    config = copy.deepcopy(cur_config)

    if task_config_path is not None:
        pass_args: List[str] = [
            f"{k}={v}" for k, v in task_config_args.items()
        ]
        task_config = habitat.get_config(
            osp.join(TASK_CONFIGS_DIR, task_config_path + ".yaml"),
            pass_args,
        )

        with habitat.config.read_write(config):
            config = OmegaConf.merge(  # type:ignore
                config, task_config.habitat.task
            )
            # config.merge_from_other_cfg(task_config.habitat.task)
            # Putting back the values from TASK_IGNORE_KEYS :
            for k in TASK_IGNORE_KEYS:
                config[k] = cur_config[k]
            # New task should not recreate any sensors
            config.measurements = {}
            config.lab_sensors = {}
    task = task_cls(config=config, dataset=cur_dataset, sim=cur_env._sim)

    assert isinstance(
        task, RearrangeTask
    ), f"Subtask must be a Rearrange Task and not {type(task)}"
    task.set_args(**task_kwargs)
    task.set_sim_reset(should_super_reset)
    task.reset(episode)
    return task
