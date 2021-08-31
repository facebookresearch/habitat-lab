#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import Any, Dict, List, Optional, Tuple

import attr

import habitat_sim.utils.datasets_download as data_downloader
from habitat.config import Config
from habitat.core.dataset import Episode
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.utils import DatasetFloatJSONEncoder
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat.datasets.utils import check_and_gen_physics_config
import numpy as np

@attr.s(auto_attribs=True, kw_only=True)
class RearrangeEpisode(Episode):
    ao_states: Dict[
        str, Tuple[int, float]
    ]  # articulated object states: instance_handle -> (link, state)
    rigid_objs: List[
        Tuple[str, np.array]
    ]  # list of objects, each with (handle, transform)
    targets: Dict[str, np.array]  # instance_name -> target_transform
    markers: Dict[str, Tuple[str, Tuple]] = {}  # marker name -> (type, params)


@registry.register_dataset(name="RearrangeDataset-v0")
class RearrangeDatasetV0(PointNavDatasetV1):
    r"""Class inherited from PointNavDataset that loads Rearrangement dataset."""
    episodes: List[RearrangeEpisode] = []  # type: ignore
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

    def to_json(self) -> str:
        result = DatasetFloatJSONEncoder().encode(self)
        return result

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config

        if config and not self.check_config_paths_exist(config):
            logger.info(
                "Rearrange task assets are not downloaded locally, downloading and extracting now..."
            )
            data_downloader.main(
                ["--uids", "rearrange_task_assets", "--no-replace"]
            )
            logger.info("Downloaded and extracted the data.")

        check_and_gen_physics_config()

        super().__init__(config)

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)

        for i, episode in enumerate(deserialized["episodes"]):
            rearrangement_episode = RearrangeEpisode(**episode)
            rearrangement_episode.episode_id = str(i)

            self.episodes.append(rearrangement_episode)
