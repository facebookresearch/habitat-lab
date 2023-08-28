#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os
import pickle
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from habitat.config import read_write
from habitat.core.dataset import ALL_SCENES_MASK, Dataset
from habitat.core.registry import registry
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    ShortestPathPoint,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


CONTENT_SCENES_PATH_FIELD = "content_scenes_path"
DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"


@registry.register_dataset(name="PointNav-v1")
class PointNavDatasetV1(Dataset):
    r"""Class inherited from Dataset that loads Point Navigation dataset."""

    episodes: List[NavigationEpisode]
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

    @staticmethod
    def check_config_paths_exist(config: "DictConfig") -> bool:
        return os.path.exists(
            config.data_path.format(split=config.split)
        ) and os.path.exists(config.scenes_dir)

    @classmethod
    def get_scenes_to_load(cls, config: "DictConfig") -> List[str]:
        r"""Return list of scene ids for which dataset has separate files with
        episodes.
        """
        dataset_dir = os.path.dirname(
            config.data_path.format(split=config.split)
        )
        if not cls.check_config_paths_exist(config):
            raise FileNotFoundError(
                f"Could not find dataset file `{dataset_dir}`"
            )

        cfg = config.copy()
        with read_write(cfg):
            cfg.content_scenes = []
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
                cfg.content_scenes = [ALL_SCENES_MASK]
                dataset = cls(cfg)
                return list(map(cls.scene_from_scene_path, dataset.scene_ids))

    @staticmethod
    def _get_scenes_from_folder(
        content_scenes_path: str, dataset_dir: str
    ) -> List[str]:
        scenes: List[str] = []
        content_dir = content_scenes_path.split("{scene}")[0]
        scene_dataset_ext = content_scenes_path.split("{scene}")[1]
        content_dir = content_dir.format(data_path=dataset_dir)
        if not os.path.exists(content_dir):
            return scenes

        for filename in os.listdir(content_dir):
            if filename.endswith(scene_dataset_ext):
                scene = filename[: -len(scene_dataset_ext)]
                scenes.append(scene)
        scenes.sort()
        return scenes

    def _load_from_file(self, fname: str, scenes_dir: str) -> None:
        """
        Load the data from a file into `self.episodes`. This can load `.pickle`
        or `.json.gz` file formats.
        """

        if fname.endswith(".pickle"):
            # NOTE: not implemented for pointnav
            with open(fname, "rb") as f:
                self.from_binary(pickle.load(f), scenes_dir=scenes_dir)
        else:
            with gzip.open(fname, "rt") as f:
                self.from_json(f.read(), scenes_dir=scenes_dir)

    def __init__(self, config: Optional["DictConfig"] = None) -> None:
        self.episodes = []

        if config is None:
            return

        datasetfile_path = config.data_path.format(split=config.split)

        self._load_from_file(datasetfile_path, config.scenes_dir)

        # Read separate file for each scene
        dataset_dir = os.path.dirname(datasetfile_path)
        has_individual_scene_files = os.path.exists(
            self.content_scenes_path.split("{scene}")[0].format(
                data_path=dataset_dir
            )
        )
        if has_individual_scene_files:
            scenes = config.content_scenes
            if ALL_SCENES_MASK in scenes:
                scenes = self._get_scenes_from_folder(
                    content_scenes_path=self.content_scenes_path,
                    dataset_dir=dataset_dir,
                )

            for scene in scenes:
                scene_filename = self.content_scenes_path.format(
                    data_path=dataset_dir, scene=scene
                )

                self._load_from_file(scene_filename, config.scenes_dir)

        else:
            self.episodes = list(
                filter(self.build_content_scenes_filter(config), self.episodes)
            )

    def to_binary(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def from_binary(
        self, data_dict: Dict[str, Any], scenes_dir: Optional[str] = None
    ) -> None:
        raise NotImplementedError()

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        for episode in deserialized["episodes"]:
            episode = NavigationEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            for g_index, goal in enumerate(episode.goals):
                episode.goals[g_index] = NavigationGoal(**goal)
            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        path[p_index] = ShortestPathPoint(**point)
            self.episodes.append(episode)
