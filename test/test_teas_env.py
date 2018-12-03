import os

import numpy as np

import teas
from teas.config.experiments.esp_nav import esp_nav_cfg


def test_teas_env():
    config = esp_nav_cfg()
    config.scene = os.environ.get('ESP_TEST_SCENE',
                                  'data/esp/test/test.glb')
    assert os.path.exists(config.scene), \
        "Please download ESP test data to data/esp/test/."
    config.freeze()
    nav = teas.make_task('EspNav-v0', config=config)
    obj, env = next(nav.episodes())

    obs = env.reset()
    done = False
    for _ in range(config.max_episode_steps):
        obs, rew, done, info = env.step(np.random.randint(3))

    # check for steps limit on environment
    assert done is True
    env.close()


if __name__ == '__main__':
    test_teas_env()
