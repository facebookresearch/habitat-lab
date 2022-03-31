#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
from typing import Any, List, Optional

from gym.envs.registration import register, registry

import habitat
import habitat.utils.env_utils
from habitat.core.environments import get_env_class
from habitat.config.default import _C, Config
from habitat_baselines.utils.gym_adapter import HabGymWrapper
from habitat_baselines.utils.render_wrapper import HabRenderWrapper

HABLAB_INSTALL_PATH = "HABLAB_BASE_CFG_PATH"

base_dir = os.environ.get(
    HABLAB_INSTALL_PATH,
    osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))),
)

gym_task_config_dir = osp.join(base_dir, "configs/tasks/rearrange/")


def _get_gym_name(cfg:Config) -> Optional[str]:
    if "GYM" in cfg:
        if "AUTO_NAME" in cfg["GYM"]:
            return cfg["GYM"]["AUTO_NAME"]
    return None

def _get_env_name(cfg:Config) -> Optional[str]:
    if "GYM" in cfg:
        if "AUTO_NAME" in cfg["GYM"]:
            return cfg["GYM"]["CLASS_NAME"]
    return None


def _make_habitat_gym_env(
    cfg_file_path: str,
    override_options: List[Any] = None,
    use_render_mode: bool = False,
):
    if override_options is None:
        override_options = []

    config = habitat.get_config(cfg_file_path)

    sensors = config["SIMULATOR"]["AGENT_0"]["SENSORS"]

    if use_render_mode:
        override_options.extend(
            [
                "TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS",
                [*sensors, "THIRD_RGB_SENSOR"],
            ]
        )

    env_class_name = _get_env_name(config)
    env_class = get_env_class(env_class_name)

    env = habitat.utils.env_utils.make_env_fn(
        env_class=env_class, config=config
    )
    env = HabGymWrapper(env)
    if use_render_mode:
        env = HabRenderWrapper(env)

    return env


def try_register(id_name, entry_point, kwargs):
    if id_name in registry.env_specs:
        return
    register(
        id_name,
        entry_point=entry_point,
        kwargs=kwargs,
    )


if "HabitatGym-v0" not in registry.env_specs:
    # Generic supporting general configs
    try_register(
        id_name="HabitatGym-v0",
        entry_point="habitat.utils.gym_definitions:_make_habitat_gym_env",
        kwargs={},
    )

    try_register(
        id_name="HabitatGymRender-v0",
        entry_point="habitat.utils.gym_definitions:_make_habitat_gym_env",
        kwargs={"use_render_mode": True},
    )

    gym_template_handle = "HabitatGym%s-v0"
    render_gym_template_handle = "HabitatGymRender%s-v0"
    for fname in os.listdir(gym_task_config_dir):
        full_path = osp.join(gym_task_config_dir, fname)
        if not fname.endswith(".yaml"):
            continue
        cfg_data = habitat.get_config(full_path)
        gym_name = _get_gym_name(cfg_data)
        if gym_name is not None:
            # Register this environment name with this config
            try_register(
                id_name=gym_template_handle % gym_name,
                entry_point="habitat.utils.gym_definitions:_make_habitat_gym_env",
                kwargs={"cfg_file_path": full_path},
            )

            try_register(
                id_name=render_gym_template_handle
                % gym_name,
                entry_point="habitat.utils.gym_definitions:_make_habitat_gym_env",
                kwargs={"cfg_file_path": full_path, "use_render_mode": True},
            )
