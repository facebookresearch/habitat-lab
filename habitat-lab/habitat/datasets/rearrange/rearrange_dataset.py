#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import attr
import numpy as np

from habitat.core.dataset import Episode
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
    :property name_to_receptacle: Map ManagedObject instance handles to containing Receptacle unique_names.
    """

    ao_states: Dict[str, Dict[int, float]]
    rigid_objs: List[Tuple[str, np.ndarray]]
    targets: Dict[str, np.ndarray]
    markers: List[Dict[str, Any]] = []
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
            raise ValueError(
                f"Requested RearrangeDataset config paths '{config.data_path.format(split=config.split)}' or '{config.scenes_dir}' are not downloaded locally. Aborting."
            )

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

    def to_binary(self) -> Dict[str, Any]:
        """
        Serialize the dataset to a pickle compatible Dict.
        """

        def access_idx(k, name_to_idx):
            if len(name_to_idx) == 0:
                name_to_idx[k] = 0
            if k not in name_to_idx:
                name_to_idx[k] = max(name_to_idx.values()) + 1
            return name_to_idx[k]

        def encode_name_dict(d, name_to_idx):
            ret_d = {}
            for k, v in d.items():
                ret_d[access_idx(k, name_to_idx)] = v
            return ret_d

        all_transforms: List[Any] = []
        name_to_idx: Dict[str, int] = {}
        all_eps = []

        for ep in self.episodes:
            new_ep_data = attr.asdict(ep)
            rigid_objs = []
            for name, T in ep.rigid_objs:
                rigid_objs.append(
                    [access_idx(name, name_to_idx), len(all_transforms)]
                )
                all_transforms.append(T)

            name_to_recep = []
            for name, recep in ep.name_to_receptacle.items():
                name_to_recep.append(
                    [
                        access_idx(name, name_to_idx),
                        access_idx(recep, name_to_idx),
                    ]
                )
            new_ep_data["rigid_objs"] = np.array(rigid_objs)
            new_ep_data["ao_states"] = encode_name_dict(
                ep.ao_states, name_to_idx
            )
            new_ep_data["name_to_receptacle"] = np.array(name_to_recep)
            new_ep_data["additional_obj_config_paths"] = list(
                new_ep_data["additional_obj_config_paths"]
            )
            del new_ep_data["_shortest_path_cache"]

            new_markers = []
            for marker_data in ep.markers:
                new_markers.append(
                    [
                        access_idx(marker_data["name"], name_to_idx),
                        access_idx(marker_data["type"], name_to_idx),
                        np.array(marker_data["params"]["offset"]),
                        access_idx(marker_data["params"]["link"], name_to_idx),
                        access_idx(
                            marker_data["params"]["object"], name_to_idx
                        ),
                    ]
                )

            new_ep_data["markers"] = new_markers

            all_eps.append(new_ep_data)

        idx_to_name = {}
        for k, v in name_to_idx.items():
            # idx_to_name should define a 1-1 mapping between the name and the
            # name index.
            assert v not in idx_to_name
            idx_to_name[v] = k

        return {
            "all_transforms": np.array(all_transforms),
            "idx_to_name": idx_to_name,
            "all_eps": all_eps,
        }

    def from_binary(
        self, data_dict: Dict[str, Any], scenes_dir: Optional[str] = None
    ) -> None:
        """
        Load the dataset from a pickle compatible Dict.
        """
        all_T = data_dict["all_transforms"]
        idx_to_name = data_dict["idx_to_name"]
        for i, ep in enumerate(data_dict["all_eps"]):
            ep["rigid_objs"] = [
                [idx_to_name[ni], all_T[ti]] for ni, ti in ep["rigid_objs"]
            ]
            ep["ao_states"] = {
                idx_to_name[ni]: v for ni, v in ep["ao_states"].items()
            }
            ep["name_to_receptacle"] = {
                idx_to_name[k]: idx_to_name[v]
                for k, v in ep["name_to_receptacle"]
            }

            new_markers = []
            for name, mtype, offset, link, obj in ep["markers"]:
                new_markers.append(
                    {
                        "name": idx_to_name[name],
                        "type": idx_to_name[mtype],
                        "params": {
                            "offset": offset,
                            "link": idx_to_name[link],
                            "object": idx_to_name[obj],
                        },
                    }
                )
            ep["markers"] = new_markers

            rearrangement_episode = RearrangeEpisode(**ep)
            rearrangement_episode.episode_id = str(i)
            self.episodes.append(rearrangement_episode)
