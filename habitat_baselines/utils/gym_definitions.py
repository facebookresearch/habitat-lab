#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
from typing import Any, List

from gym.envs.registration import register

import habitat
import habitat_baselines.utils.env_utils
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.config.default import _C
from habitat_baselines.config.default import get_config as baselines_get_config
from habitat_baselines.utils.gym_adapter import HabGymWrapper
from habitat_baselines.utils.render_wrapper import HabRenderWrapper

GYM_AUTO_NAME_KEY = "GYM_AUTO_NAME"
HABLAB_INSTALL_PATH = "HABLAB_BASE_CFG_PATH"

base_dir = os.environ.get(
    HABLAB_INSTALL_PATH,
    osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))),
)


def _get_config_no_base_task_load(cfg_file_path):
    """
    Load in a Habitat Baselines config file without loading the BASE_TASK_CONFIG_PATH file. The purpose of this is to load the config even if the BASE_TASK_CONFIG_PATH does not exist.
    """
    cfg_data = _C.clone()
    cfg_data.merge_from_file(cfg_file_path)
    return cfg_data


def _make_habitat_gym_env(
    cfg_file_path: str,
    override_options: List[Any] = None,
    use_render_mode: bool = False,
):
    if override_options is None:
        override_options = []

    cfg_data = _get_config_no_base_task_load(cfg_file_path)
    task_cfg_path = osp.join(base_dir, cfg_data["BASE_TASK_CONFIG_PATH"])

    override_options.extend(["BASE_TASK_CONFIG_PATH", task_cfg_path])

    task_cfg_data = habitat.get_config(task_cfg_path)
    sensors = task_cfg_data["SIMULATOR"]["AGENT_0"]["SENSORS"]

    if use_render_mode:
        override_options.extend(
            [
                "TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS",
                [*sensors, "THIRD_RGB_SENSOR"],
            ]
        )

    config = baselines_get_config(cfg_file_path, override_options)
    env_class = get_env_class(config.ENV_NAME)

    env = habitat_baselines.utils.env_utils.make_env_fn(
        env_class=env_class, config=config
    )
    env = HabGymWrapper(env)
    if use_render_mode:
        env = HabRenderWrapper(env)

    return env


# Generic supporting general configs
register(
    id="HabitatGym-v0",
    entry_point="habitat_baselines.utils.gym_definitions:_make_habitat_gym_env",
)

register(
    id="HabitatGymRender-v0",
    entry_point="habitat_baselines.utils.gym_definitions:_make_habitat_gym_env",
    kwargs={"use_render_mode": True},
)


hab_baselines_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
rearrange_configs_dir = osp.join(hab_baselines_dir, "config/rearrange/")
gym_template_handle = "HabitatGym%s-v0"
render_gym_template_handle = "HabitatGymRender%s-v0"
for fname in os.listdir(rearrange_configs_dir):
    full_path = osp.join(rearrange_configs_dir, fname)
    if not fname.endswith(".yaml"):
        continue
    cfg_data = _get_config_no_base_task_load(full_path)
    if GYM_AUTO_NAME_KEY in cfg_data:
        # Register this environment name with this config
        register(
            id=gym_template_handle % cfg_data[GYM_AUTO_NAME_KEY],
            entry_point="habitat_baselines.utils.gym_definitions:_make_habitat_gym_env",
            kwargs={"cfg_file_path": full_path},
        )

        register(
            id=render_gym_template_handle % cfg_data[GYM_AUTO_NAME_KEY],
            entry_point="habitat_baselines.utils.gym_definitions:_make_habitat_gym_env",
            kwargs={"cfg_file_path": full_path, "use_render_mode": True},
        )
