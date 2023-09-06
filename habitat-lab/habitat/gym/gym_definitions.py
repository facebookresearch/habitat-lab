#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
from typing import TYPE_CHECKING, Any, List, Optional

import gym
from gym.envs.registration import register, registry

from habitat import get_config, read_write
from habitat.config.default import _HABITAT_CFG_DIR, register_configs
from habitat.config.default_structured_configs import (
    SimulatorSensorConfig,
    ThirdRGBSensorConfig,
)
from habitat.core.environments import get_env_class
from habitat.utils.env_utils import make_env_fn

if TYPE_CHECKING:
    from omegaconf import DictConfig


PRE_REGISTERED_GYM_TASKS = {
    "CloseFridge": "benchmark/rearrange/skills/close_fridge.yaml",
    "Pick": "benchmark/rearrange/skills/pick.yaml",
    "NavToObj": "benchmark/rearrange/skills/nav_to_obj.yaml",
    "ReachState": "benchmark/rearrange/skills/reach_state.yaml",
    "CloseCab": "benchmark/rearrange/skills/close_cab.yaml",
    "SetTable": "benchmark/rearrange/multi_task/set_table.yaml",
    "OpenCab": "benchmark/rearrange/skills/open_cab.yaml",
    "Place": "benchmark/rearrange/skills/place.yaml",
    "Rearrange": "benchmark/rearrange/multi_task/rearrange.yaml",
    "PrepareGroceries": "benchmark/rearrange/multi_task/prepare_groceries.yaml",
    "RearrangeEasy": "benchmark/rearrange/multi_task/rearrange_easy.yaml",
    "OpenFridge": "benchmark/rearrange/skills/open_fridge.yaml",
    "TidyHouse": "benchmark/rearrange/multi_task/tidy_house.yaml",
}


def _get_env_name(cfg: "DictConfig") -> Optional[str]:
    if "habitat" in cfg:
        cfg = cfg.habitat
    return cfg["env_task"]


def make_gym_from_config(config: "DictConfig", dataset=None) -> gym.Env:
    """
    From a habitat-lab or habitat-baseline config, create the associated gym environment.
    """
    if "habitat" in config:
        config = config.habitat
    env_class_name = _get_env_name(config)
    env_class = get_env_class(env_class_name)
    assert (
        env_class is not None
    ), f"No environment class with name `{env_class_name}` was found, you need to specify a valid one with env_task"
    return make_env_fn(env_class=env_class, config=config, dataset=dataset)


def _add_sim_sensor_to_config(
    config: "DictConfig", sensor: SimulatorSensorConfig
):
    with read_write(config):
        sim_config = config.habitat.simulator
        default_agent_name = sim_config.agents_order[
            sim_config.default_agent_id
        ]
        default_agent = sim_config.agents[default_agent_name]
        if len(sim_config.agents) == 1:
            default_agent.sim_sensors.update({"third_rgb_sensor": sensor})
        else:
            default_agent.sim_sensors.update(
                {"default_agent_third_rgb_sensor": sensor}
            )


def _make_habitat_gym_env(
    cfg_file_path: str,
    override_options: List[Any] = None,
    use_render_mode: bool = False,
) -> gym.Env:
    if override_options is None:
        override_options = []

    config = get_config(cfg_file_path, overrides=override_options)
    if use_render_mode:
        _add_sim_sensor_to_config(config, ThirdRGBSensorConfig())
    env = make_gym_from_config(config)
    return env


def _try_register(id_name, entry_point, kwargs):
    if id_name in registry.env_specs:
        return
    register(
        id_name,
        entry_point=entry_point,
        kwargs=kwargs,
    )


if "Habitat-v0" not in registry.env_specs:
    register_configs()
    # Generic supporting general configs
    _try_register(
        id_name="Habitat-v0",
        entry_point="habitat.gym.gym_definitions:_make_habitat_gym_env",
        kwargs={},
    )

    _try_register(
        id_name="HabitatRender-v0",
        entry_point="habitat.gym.gym_definitions:_make_habitat_gym_env",
        kwargs={"use_render_mode": True},
    )

    hab_baselines_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    gym_template_handle = "Habitat%s-v0"
    render_gym_template_handle = "HabitatRender%s-v0"

    for gym_name, file_name in PRE_REGISTERED_GYM_TASKS.items():
        # Register this environment name with this config
        full_path = osp.join(_HABITAT_CFG_DIR, file_name)
        _try_register(
            id_name=gym_template_handle % gym_name,
            entry_point="habitat.gym.gym_definitions:_make_habitat_gym_env",
            kwargs={"cfg_file_path": full_path},
        )
        # print(gym_template_handle % gym_name)

        _try_register(
            id_name=render_gym_template_handle % gym_name,
            entry_point="habitat.gym.gym_definitions:_make_habitat_gym_env",
            kwargs={"cfg_file_path": full_path, "use_render_mode": True},
        )
