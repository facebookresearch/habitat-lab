#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import Any, Dict, List, Optional
import os

import attr

import habitat_sim.utils.datasets_download as data_downloader
from habitat.config import Config
from habitat.core.dataset import Episode, Dataset
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.utils import DatasetFloatJSONEncoder
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat.datasets.utils import check_and_gen_physics_config


@attr.s(auto_attribs=True, kw_only=True)
class AntV2Episode(Episode):
    pass

@registry.register_dataset(name="AntV2Dataset-v0")
class AntV2DatasetV0(Dataset):
    r"""Class inherited from PointNavDataset that loads Ant dataset."""
    episodes: List[AntV2Episode] = []  # type: ignore"

    def to_json(self) -> str:
        result = DatasetFloatJSONEncoder().encode(self)
        return result

    @staticmethod
    def check_config_paths_exist(config: Config) -> bool:
        return os.path.exists(
            config.DATA_PATH.format(split=config.SPLIT)
        ) and os.path.exists(config.SCENES_DIR)
    
    @classmethod
    def get_scenes_to_load(cls, config: Config) -> List[str]:
        r"""Return list of scene ids for which dataset has separate files with
        episodes.
        """
        print("CONFIG:::")

        print(config)
        print("CONFIG:::")
        """assert cls.check_config_paths_exist(config)
        dataset_dir = os.path.dirname(
            config.DATA_PATH.format(split=config.SPLIT)
        )

        cfg = config.clone()
        cfg.defrost()
        cfg.CONTENT_SCENES = []
        dataset = cls(cfg)
        has_individual_scene_files = os.path.exists(
            dataset.content_scenes_path.split("{scene}")[0].format(
                data_path=dataset_dir
            )
        )
        if has_individual_scene_files:
            return cls._get_scenes_from_folder(
                content_scenes_path=dataset.content_scenes_path,
                dataset_dir=dataset_dir,
            )
        else:
            # Load the full dataset, things are not split into separate files
            cfg.CONTENT_SCENES = [ALL_SCENES_MASK]
            dataset = cls(cfg)
            return list(map(cls.scene_from_scene_path, dataset.scene_ids))"""
        cfg = config.clone()
        cfg.defrost()
        cfg.CONTENT_SCENES = ["NONE"]
        dataset = cls(cfg)
        return list(map(cls.scene_from_scene_path, dataset.scene_ids))

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config

        check_and_gen_physics_config()

        super().__init__()

        ant_v2_episode = AntV2Episode(start_position = [0,0,0], start_rotation = [0,0,0,1], episode_id ="0", scene_id = "NONE")
        self.episodes.append(ant_v2_episode)

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        pass
        
