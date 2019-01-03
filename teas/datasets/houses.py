import json
from random import shuffle

import teas


# TODO(akadian): Remove the below legacy code
class SuncgDataset(teas.Dataset):
    def __init__(self, config, shuffle_order=False):
        with open(config.data_path, 'r') as f:
            data_json = json.load(f)
        self.split = config.split
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
