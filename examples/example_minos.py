import pickle

from PIL import Image

import teas
from teas.config.experiments.minos_eqa import minos_eqa_cfg


MAX_EPISODE_STEPS = 10
NUM_EPISODES = 5


def main():
    # TODO(akadian): Add descriptive comments for example
    minos_eqa_cfg.freeze()
    eqa = teas.make_task('MinosEqa-v0', config=minos_eqa_cfg)
    
    images = []
    
    for ep_i, (ques, ans, env) in enumerate(eqa.episodes()):
        print("Episode: {}".format(ep_i))
        print("Question:", ques)
        print("Answer:", ans)
        obs = env.reset()
        images.append(Image.fromarray(obs['rgb'], 'RGBA'))
        for step in range(MAX_EPISODE_STEPS):
            action = env.action_space.sample()
            obs, rew, done, info = env.step(action)
            img = obs['rgb']
            img = Image.fromarray(img, 'RGBA')
            images.append(img)
        if ep_i > NUM_EPISODES:
            # example setup
            break
    
    eqa.close()
    
    with open('res.pkl', 'wb') as f:
        pickle.dump(images, f)


if __name__ == '__main__':
    main()
