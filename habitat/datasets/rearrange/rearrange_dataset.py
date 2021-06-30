#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import Any, Dict, List, Optional

import attr

from habitat.config import Config
from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.core.utils import DatasetFloatJSONEncoder
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1


@attr.s(auto_attribs=True, kw_only=True)
class RearrangeEpisode(Episode):
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


@registry.register_dataset(name="RearrangeDataset-v0")
class RearrangeDatasetV0(PointNavDatasetV1):
    r"""Class inherited from PointNavDataset that loads Rearrangement dataset."""
    episodes: List[RearrangeEpisode] = []  # type: ignore
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

    def to_json(self) -> str:
        result = DatasetFloatJSONEncoder().encode(self)
        return result

    def __init__(self, config: Optional[Config] = None) -> None:
        super().__init__(config)
        self.config = config

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)

        for i, episode in enumerate(deserialized["episodes"]):
            rearrangement_episode = RearrangeEpisode(**episode)
            rearrangement_episode.episode_id = str(i)
            (
                rearrangement_episode.scene_id.replace(
                    "data/scene_datasets/", "data/replica_cad/stages/Stage_"
                )[:-7]
                + ".glb"
            )
            self.episodes.append(rearrangement_episode)
