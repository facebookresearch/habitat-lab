import os
import shutil

import numpy as np
import cv2

import habitat
from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video

def ant_environment_example():
    config = habitat.get_config(config_paths="configs/tasks/ant-v2.yaml")
    config.defrost()
    config.freeze()
    with habitat.Env(config=config) as env:
        env.reset()
        while not env.episode_over:
            action = env.action_space.sample()
            obs = env.step(action)
            keystroke = cv2.waitKey(0)

            if keystroke == 27:
                break
            print(obs["ant_observation_space_sensor"])
            cv2.imshow("RGB", obs["robot_third_rgb"])

        env.reset()


def main():
    ant_environment_example()


if __name__ == "__main__":
    main()

