#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from omegaconf import DictConfig, ListConfig

from habitat.config.default_structured_configs import (
    HabitatConfigPlugin,
    register_hydra_plugin,
)
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesConfigPlugin,
)


class HabitatHitlConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(
            provider="habitat",
            path="pkg://habitat_hitl/config",
        )


def register_hydra_plugins():
    register_hydra_plugin(HabitatConfigPlugin)
    register_hydra_plugin(HabitatBaselinesConfigPlugin)
    register_hydra_plugin(HabitatHitlConfigPlugin)


class ConfigObject:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def omegaconf_to_object(cfg):
    """
    Recursively convert an OmegaConf configuration to a Python object. The motivation
    here is that accessing config fields at runtime is ~50x slower than accessing pure
    Python object fields.
    If the configuration is a DictConfig, it creates a ConfigObject with the dictionary items.
    If the configuration is a ListConfig, it creates a list with the list items.
    In both cases, it recursively converts the items to objects.
    If the configuration is not a DictConfig or a ListConfig, it returns the configuration value as is.
    Args:
        cfg (omegaconf.Config): The OmegaConf configuration to convert.
    Returns:
        A Python object that represents the given OmegaConf configuration. This could be a ConfigObject,
        a list, or a basic data type, depending on the structure of the configuration.
    """
    if isinstance(cfg, DictConfig):
        return ConfigObject(
            **{k: omegaconf_to_object(v) for k, v in cfg.items()}  # type: ignore
        )
    elif isinstance(cfg, ListConfig):
        return [omegaconf_to_object(v) for v in cfg]
    else:
        return cfg
