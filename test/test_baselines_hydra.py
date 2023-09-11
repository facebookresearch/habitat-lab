# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

from hydra import compose, initialize
from omegaconf import OmegaConf

# NOTE: import required to register structured configs
import habitat_baselines.config.default_structured_configs  # noqa: F401
from habitat.config.default_structured_configs import (
    HabitatConfigPlugin,
    register_hydra_plugin,
)
from habitat_baselines.run import execute_exp

if TYPE_CHECKING:
    from omegaconf import DictConfig


def my_app_compose_api() -> "DictConfig":
    # initialize the Hydra subsystem.
    # This is needed for apps that cannot have
    # a standard @hydra.main() entry point
    with initialize(version_base=None):
        cfg = compose(
            overrides=[
                "+habitat_baselines=habitat_baselines_rl_config_base",
                "habitat_baselines.num_updates=2",
                "habitat_baselines.num_environments=4",
                "habitat_baselines.total_num_steps=-1.0",
                "habitat_baselines.rl.policy.main_agent.action_distribution_type=gaussian",
                "+benchmark/rearrange/skills=pick",
                "habitat.simulator.agents_order=[main_agent]",
            ]
        )

    # OmegaConf.set_readonly(cfg, True)
    return cfg


def test_hydra_configs():
    # Manually register the habitat configs so we can use the task configs (`+habitat=config"`)
    register_hydra_plugin(HabitatConfigPlugin)

    cfg = my_app_compose_api()
    print(OmegaConf.to_yaml(cfg))
    execute_exp(cfg, "train")


if __name__ == "__main__":
    test_hydra_configs()
