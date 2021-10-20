#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import Any, Dict, List, Optional

import attr

import habitat_sim.utils.datasets_download as data_downloader
from habitat.config import Config
from habitat.core.dataset import Episode
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.utils import DatasetFloatJSONEncoder
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat.datasets.utils import check_and_gen_physics_config


@attr.s(auto_attribs=True, kw_only=True)
class AntV2Episode(Episode):
    art_objs: List[List[Any]]
    static_objs: List[List[Any]]
    targets: List[List[Any]]
    fixed_base: bool
    art_states: List[Any]
    nav_mesh_path: str
    scene_config_path: str
    allowed_region: List[Any] = []
    markers: List[Dict[str, Any]] = []
    force_spawn_pos: List = None


@registry.register_dataset(name="AntV2Dataset-v0")
class AntV2DatasetV0(PointNavDatasetV1):
    r"""Class inherited from PointNavDataset that loads Ant dataset."""
    episodes: List[AntV2Episode] = []  # type: ignore
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

    def to_json(self) -> str:
        result = DatasetFloatJSONEncoder().encode(self)
        return result

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config

        ant_v2_episode = AntV2Episode()
        ant_v2_episode.art_objs = [[]]
        episodes.append(ant_v2_episode)


        check_and_gen_physics_config()

        super().__init__(config)

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        pass
        
