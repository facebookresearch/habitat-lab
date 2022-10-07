from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

# NOTE: import required to register structured configs
# import habitat.config.default_structured_configs
import habitat_baselines.config.default_structured_configs


def my_app_compose_api() -> DictConfig:
    # initialize the Hydra subsystem.
    # This is needed for apps that cannot have
    # a standard @hydra.main() entry point
    with initialize(version_base=None):

        cfg = compose(
            overrides=[
                # "+habitat=config",
                # "+habitat.task.actions.turn_left=turn_left",
                # "+habitat.task.actions.turn_right=turn_right",
                # "+habitat.task.actions.stop=stop",
                # "+habitat.task.actions.move_forward=move_forward",

                "+habitat_baselines=habitat_baselines_config_base",
                "+habitat_baselines/rl/auxiliary_losses/cpca=cpca_loss_base",
                "+habitat_baselines/rl/policy/obs_transforms/center_cropper=center_cropper_base"
            ]
        )

    # OmegaConf.set_readonly(cfg, True)
    return cfg


def test_hydra_configs():
    cfg = my_app_compose_api()
    print(OmegaConf.to_yaml(cfg))

    # Access interpolated values from habitat namespace
    # print(cfg.habitat_baselines.orbslam2.camera_height)
    # print(cfg.habitat_baselines.orbslam2.depth_denorm)


if __name__ == "__main__":
    test_hydra_configs()
