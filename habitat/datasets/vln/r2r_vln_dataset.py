#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from typing import List, Optional
from habitat.config import Config
from habitat.tasks.vln.vln import VLNEpisode, InstructionData
from habitat.core.dataset import Dataset
from habitat.core.registry import registry
from habitat.datasets.utils import VocabDict
from habitat.tasks.nav.nav import NavigationGoal, ShortestPathPoint
from habitat.core.logging import logger

CONTENT_SCENES_PATH_FIELD = "content_scenes_path"
DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/mp3d/"

R2R_TRAIN_EPISODES = 10837
R2R_VAL_SEEN_EPISODES = 781
R2R_VAL_UNSEEN_EPISODES = 1839

@registry.register_dataset(name="R2RVLN-v1")
class VLNDatasetV1(Dataset):
    r"""Class inherited from Dataset that loads the MatterPort3D
    Room-to-Room (R2R) dataset for Vision and Language Navigation.
    """

    episodes: List[VLNEpisode]
    instruction_vocab: VocabDict

    @staticmethod
    def check_config_paths_exist(config: Config) -> bool:
        return os.path.exists(
            config.DATA_PATH.format(split=config.SPLIT)
        ) and os.path.exists(config.SCENES_DIR)


    def __init__(self, config: Optional[Config] = None) -> None:
        self.episodes = []

        if config is None:
            return
        
        datasetfile_path = config.DATA_PATH.format(split=config.SPLIT)
        with open(datasetfile_path) as json_file:
            json_str = json.load(json_file)
        self.from_json(json_str, scenes_dir=config.SCENES_DIR)

    def from_json(
        self, deserialized: [str], scenes_dir: Optional[str] = None
    ) -> None:
        
        #Done for serialization test
        if isinstance(deserialized, str):
            deserialized = json.loads(deserialized)
            self.instruction_vocab = VocabDict(word_list=deserialized["instruction_vocab"]["word_list"])
        else:
            self.instruction_vocab = VocabDict(word_list=deserialized["instruction_vocab"])


        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]
        for episode in deserialized["episodes"]:
            episode = VLNEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id, episode.scene_id + ".glb")

            episode.instruction = InstructionData(**episode.instruction)
            for g_index, goal in enumerate(episode.goals):
                episode.goals[g_index] = NavigationGoal(**goal)
            self.episodes.append(episode)
