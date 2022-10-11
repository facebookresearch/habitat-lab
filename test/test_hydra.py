from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

import habitat

# NOTE: import required to register structured configs
import habitat.config.default_structured_configs


def my_app_compose_api() -> DictConfig:
    # initialize the Hydra subsystem.
    # This is needed for apps that cannot have
    # a standard @hydra.main() entry point
    with initialize(version_base=None):

        cfg = compose(
            overrides=[
                "+habitat=config",
                "+habitat.task.actions.turn_left=turn_left",
                "+habitat.task.actions.turn_right=turn_right",
                "+habitat.task.actions.stop=stop",
                "+habitat.task.actions.move_forward=move_forward",
            ]
        )

    OmegaConf.set_readonly(cfg, True)

    return cfg


def run_example_pointnav_old_config():
    config = habitat.get_config()
    with habitat.Env(config=config) as env:
        print("Environment creation successful")
        observations = env.reset()  # noqa: F841

        print("Agent stepping around inside environment.")
        count_steps = 0
        while not env.episode_over:
            observations = env.step(env.action_space.sample())  # noqa: F841
            count_steps += 1
        print("Episode finished after {} steps.".format(count_steps))


def run_example_pointnav(config):
    # config = OmegaConf.to_container(config)
    with habitat.Env(config=config) as env:
        print("Environment creation successful")
        observations = env.reset()  # noqa: F841

        print("Agent stepping around inside environment.")
        count_steps = 0
        while not env.episode_over:
            observations = env.step(env.action_space.sample())  # noqa: F841
            count_steps += 1
        print("Episode finished after {} steps.".format(count_steps))


def test_hydra_configs():
    cfg = my_app_compose_api()
    run_example_pointnav(cfg)
    # print(OmegaConf.to_yaml(cfg))


def test_yacs_config():
    run_example_pointnav_old_config()
