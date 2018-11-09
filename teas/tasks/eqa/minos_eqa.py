import teas
from teas.datasets import make_dataset
from teas.wrappers.timelimit import EnvTimeLimit


class MinosEqaTask(teas.EmbodiedTask):
    def __init__(self, config):
        self._dataset = make_dataset('Suncg-v0', config=config.dataset)
        self._env = EnvTimeLimit(
            teas.TeasEnv(config=config.env),
            max_episode_seconds=config.env.max_episode_seconds,
            max_episode_steps=config.env.max_episode_steps)
        self.seed(config.seed)
    
    def episodes(self):
        for i in range(len(self._dataset)):
            hid, ques, ans = self._dataset[i]
            self._env.reconfigure(hid)
            yield ques, ans, self._env
    
    def seed(self, seed):
        self._env.seed(seed)
    
    def close(self):
        self._env.close()
