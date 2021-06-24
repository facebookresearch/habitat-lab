import json
import os.path as osp
from typing import List, Optional

import attr

from habitat.config import Config
from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.core.simulator import ShortestPathPoint
from habitat.core.utils import DatasetFloatJSONEncoder, not_none_validator
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat.tasks.nav.nav import NavigationGoal


@attr.s(auto_attribs=True, kw_only=True)
class OrpEpisode(Episode):
    art_objs: object
    static_objs: object
    targets: object
    fixed_base: bool
    art_states: object
    nav_mesh_path: str
    scene_config_path: str
    allowed_region: List = []
    markers: List = []
    force_spawn_pos: List = None


@registry.register_dataset(name="OrpDataset-v0")
class OrpDatasetV0(PointNavDatasetV1):
    r"""Class inherited from PointNavDataset that loads Rearrangement dataset."""
    episodes: List[OrpEpisode]
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

    def to_json(self) -> str:
        result = DatasetFloatJSONEncoder().encode(self)
        return result

    def __init__(self, config: Optional[Config] = None) -> None:
        super().__init__(config)

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        dir_path = osp.dirname(osp.realpath(__file__))

        for i, episode in enumerate(deserialized["episodes"]):
            rearrangement_episode = OrpEpisode(**episode)
            rearrangement_episode.episode_id = str(i)
            self.episodes.append(rearrangement_episode)


@attr.s(auto_attribs=True, kw_only=True)
class OrpNavEpisode(OrpEpisode):
    goals: List[NavigationGoal] = attr.ib(
        default=None, validator=not_none_validator
    )
    start_room: Optional[str] = None
    shortest_paths: Optional[List[List[ShortestPathPoint]]] = None


@registry.register_dataset(name="OrpNavDataset-v0")
class OrpNavDatasetV0(OrpDatasetV0):
    episodes: List[OrpNavEpisode]

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        dir_path = osp.dirname(osp.realpath(__file__))

        for i, episode in enumerate(deserialized["episodes"]):
            episode = OrpNavEpisode(**episode)
            episode.episode_id = str(i)

            for g_index, goal in enumerate(episode.goals):
                episode.goals[g_index] = NavigationGoal(**goal)

            self.episodes.append(episode)
