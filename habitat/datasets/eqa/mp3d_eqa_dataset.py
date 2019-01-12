import gzip
import json
import os
from typing import Any, List

from habitat.core.dataset import Dataset
from habitat.tasks.eqa.eqa_task import EQAEpisode, QuestionData
from habitat.tasks.nav.nav_task import (
    ObjectGoal, ShortestPathPoint
)
from yacs.config import CfgNode

EQA_MP3D_V1_VAL_EPISODE_COUNT = 1950


def get_default_mp3d_v1_config(split: str = "val"):
    config = CfgNode()
    config.name = "MP3DEQA-v1"
    config.data_path = "data/datasets/eqa/mp3d/v1/{split}.json.gz"
    config.scenes_path = "data/scene_datasets/mp3d"
    config.split = split
    return config


class Matterport3dDatasetV1(Dataset):
    r"""Class inherited from Dataset that loads Matterport3D
    Embodied Question Answering dataset.

    This class can then be used as follows::


    eqa_config.dataset = get_default_mp3d_v1_config()
    eqa = habitat.make_task(eqa_config.task_name, config=eqa_config)

    """
    episodes: List[EQAEpisode]

    @staticmethod
    def check_config_paths_exist(config: Any) -> bool:
        return os.path.exists(config.data_path.format(split=config.split)) \
               and os.path.exists(config.scenes_path)

    def __init__(self, config: Any = None) -> None:
        self.episodes = []

        if config is None:
            return

        with gzip.open(config.data_path.format(split=config.split), "rt") as f:
            self.from_json(f.read())

    def from_json(self, serialized: str) -> None:
        deserialized = json.loads(serialized)
        self.__dict__.update(deserialized)
        for ep_index, episode in enumerate(deserialized["episodes"]):
            episode = EQAEpisode(**episode)
            episode.question = QuestionData(**episode.question)
            for g_index, goal in enumerate(episode.goals):
                episode.goals[g_index] = ObjectGoal(**goal)
            for path in episode.shortest_paths:
                for p_index, point in enumerate(path):
                    path[p_index] = ShortestPathPoint(**point)
            self.episodes[ep_index] = episode
