import pickle

from PIL import Image

import teas
from teas.core.logging import logger
from teas.config.experiments.esp_nav import esp_nav_cfg


def main():
    config = esp_nav_cfg()
    config.freeze()
    nav = teas.make_task('Nav-v0', config=config)

    for ep_i, (target_object, env) in enumerate(nav.episodes()):
        images = []
        env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, rew, done, info = env.step(action)
            images.append(Image.fromarray(obs['rgb'], 'RGBA'))

        with open('res.pkl', 'wb') as f:
            pickle.dump(images, f)
            logger.info("successfully saved images")
    nav.close()


if __name__ == '__main__':
    main()
