import pickle

from PIL import Image

import teas
from teas.config.experiments.esp_nav import esp_nav_cfg


def main():
    esp_nav_cfg.freeze()
    nav = teas.make_task('EspNav-v0', config=esp_nav_cfg)
    
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
        print("successfully saved images")
    nav.close()


if __name__ == '__main__':
    main()
