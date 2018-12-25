import os

import numpy as np

import teas
from teas.config.experiments.esp_nav import esp_nav_cfg
from teas.core.logging import logger
from teas.tasks.nav.nav_task import NavigationEpisode


def test_teas_env():
    config = esp_nav_cfg()
    config.task_name = 'Nav-v0'
    assert os.path.exists(config.scene), \
        "Please download ESP test data to data/esp/test/."
    env = teas.TeasEnv(config=config, dataset=None)
    env.episodes = [NavigationEpisode(
        episode_id="0",
        scene_id=config.scene,
        start_position=[03.00611, 0.072447, -2.67867],
        start_rotation=[0, 0.163276, 0, 0.98658],
        goals=[])]

    env.reset()
    done = False
    for _ in range(config.max_episode_steps):
        obs, rew, done, info = env.step(np.random.randint(3))

    # check for steps limit on environment
    assert done is True
    env.close()
    logger.info("test_teas_env test successful")


if __name__ == '__main__':
    test_teas_env()
