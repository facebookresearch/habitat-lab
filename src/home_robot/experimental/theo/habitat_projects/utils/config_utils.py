from typing import Tuple, Optional
import json
import yaml

import habitat.config.default
from habitat.config.default import Config


def get_config(path: str, opts: Optional[list] = None) -> Tuple[Config, str]:
    """Get configuration and ensure consistency between configurations
    inherited from the task and defaults and our code's configuration.

    Arguments:
        path: path to our code's config
        opts: command line arguments overriding the config
    """
    config = Config()

    # Start with our code's config
    config.merge_from_file(path)

    # Add Habitat's default config and the base task config path specified
    # in our code's config under TASK_CONFIG
    task_config = Config()
    task_config.merge_from_other_cfg(habitat.config.default._C)
    task_config.merge_from_file(config.BASE_TASK_CONFIG_PATH)
    config.TASK_CONFIG = task_config

    # Add command line arguments
    if opts is not None:
        config.merge_from_list(opts)

    config.freeze()

    # Generate a string representation of our code's config
    config_dict = yaml.load(open(path), Loader=yaml.FullLoader)
    if opts is not None:
        for i in range(0, len(opts), 2):
            dict = config_dict
            keys = opts[i].split(".")
            if "TASK_CONFIG" in keys:
                continue
            value = opts[i + 1]
            for key in keys[:-1]:
                dict = dict[key]
            dict[keys[-1]] = value
    config_str = json.dumps(config_dict, indent=4)

    # Ensure consistency between configurations inherited from the task
    # and defaults and our code's configuration
    rgb_sensor = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR
    depth_sensor = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR
    semantic_sensor = config.TASK_CONFIG.SIMULATOR.get("SEMANTIC_SENSOR")

    frame_height = config.ENVIRONMENT.frame_height
    assert rgb_sensor.HEIGHT == depth_sensor.HEIGHT
    if semantic_sensor:
        assert rgb_sensor.HEIGHT == semantic_sensor.HEIGHT
    assert rgb_sensor.HEIGHT >= frame_height and rgb_sensor.HEIGHT % frame_height == 0

    frame_width = config.ENVIRONMENT.frame_width
    assert rgb_sensor.WIDTH == depth_sensor.WIDTH
    if semantic_sensor:
        assert rgb_sensor.WIDTH == semantic_sensor.WIDTH
    assert rgb_sensor.WIDTH >= frame_width and rgb_sensor.WIDTH % frame_width == 0

    camera_height = config.ENVIRONMENT.camera_height
    assert camera_height == rgb_sensor.POSITION[1]
    assert camera_height == depth_sensor.POSITION[1]
    if semantic_sensor:
        assert camera_height == semantic_sensor.POSITION[1]

    hfov = config.ENVIRONMENT.hfov
    assert hfov == rgb_sensor.HFOV
    assert hfov == depth_sensor.HFOV
    if semantic_sensor:
        assert hfov == semantic_sensor.HFOV

    assert config.ENVIRONMENT.min_depth == depth_sensor.MIN_DEPTH
    assert config.ENVIRONMENT.max_depth == depth_sensor.MAX_DEPTH
    assert config.ENVIRONMENT.turn_angle == config.TASK_CONFIG.SIMULATOR.TURN_ANGLE

    return config, config_str
