import os
import shutil

import numpy as np

import habitat
from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video



def ant_environment_example():
    print("Code is running..")
    config = habitat.get_config(config_paths="configs/tasks/ant-v2.yaml")
    config.defrost()
    config.freeze()
    print("Config loaded..")
    with habitat.Env(config=config) as env:
        print("Environent constructed..")
        env.reset()
        while not env.episode_over:
            action = env.action_space.sample()
            obs = env.step(action)
            habitat.logger.info(
                f"Action : "
                f"{action['action']}, "
                f"args: {action['action_args']}."
                f"obs: {list(obs.keys())}."
            )
        env.reset()


def main():
    ant_environment_example()


if __name__ == "__main__":
    main()

