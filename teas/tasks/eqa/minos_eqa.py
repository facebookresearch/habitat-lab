import json
from random import shuffle

import teas
from teas.datasets import make_dataset
from teas.wrappers.timelimit import EnvTimeLimit


class SuncgDataset(teas.Dataset):
    def __init__(self, path, split, shuffle_order=False):
        with open(path, 'r') as f:
            data_json = json.load(f)
        self.split = split
        houses = {}
        for hid in data_json['questions']:
            houses[hid] = data_json['questions'][hid]
        hids = data_json['splits'][self.split]
        self.data = []
        for hid in hids:
            for x in houses[hid]:
                self.data.append((hid, x['question'], x['answer']))
        if shuffle_order:
            shuffle(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)


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
