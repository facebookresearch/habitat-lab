#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import attr
import numpy as np

import habitat_sim.utils.datasets_download as data_downloader
from habitat.core.dataset import Episode
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.utils import DatasetFloatJSONEncoder
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat.datasets.utils import check_and_gen_physics_config

if TYPE_CHECKING:
    from omegaconf import DictConfig


@attr.s(auto_attribs=True, kw_only=True)
class RearrangeEpisode(Episode):
    r"""Specifies additional objects, targets, markers, and ArticulatedObject states for a particular instance of an object rearrangement task.

    :property ao_states: Lists modified ArticulatedObject states for the scene: {instance_handle -> {link, state}}
    :property rigid_objs: A list of objects to add to the scene, each with: (handle, transform)
    :property targets: Maps an object instance to a new target location for placement in the task. {instance_name -> target_transform}
    :property markers: Indicate points of interest in the scene such as grasp points like handles. {marker name -> (type, (params))}
    :property target_receptacles: The names and link indices of the receptacles containing the target objects.
    :property goal_receptacles: The names and link indices of the receptacles containing the goals.
    """
    ao_states: Dict[str, Dict[int, float]]
    rigid_objs: List[Tuple[str, np.ndarray]]
    targets: Dict[str, np.ndarray]
    markers: Dict[str, Tuple[str, Tuple]] = {}
    target_receptacles: List[Tuple[str, int]] = []
    goal_receptacles: List[Tuple[str, int]] = []
    name_to_receptacle: Dict[str, str] = {}


@registry.register_dataset(name="RearrangeDataset-v0")
class RearrangeDatasetV0(PointNavDatasetV1):
    r"""Class inherited from PointNavDataset that loads Rearrangement dataset."""
    episodes: List[RearrangeEpisode] = []  # type: ignore
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

    def to_json(self) -> str:
        result = DatasetFloatJSONEncoder().encode(self)
        return result

    def __init__(self, config: Optional["DictConfig"] = None) -> None:
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
