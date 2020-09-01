import json
import os
from typing import List, Optional

from habitat.config import Config
from habitat.core.registry import registry
from habitat.core.utils import DatasetFloatJSONEncoder
from habitat.datasets.pointnav.pointnav_dataset import (
    CONTENT_SCENES_PATH_FIELD,
    DEFAULT_SCENE_PATH_PREFIX,
    PointNavDatasetV1,
)
from habitat.tasks.rearrangement.rearrangement_task import (
    RearrangementEpisode,
    RearrangementObjectSpec,
    RearrangementSpec,
)


@registry.register_dataset(name="Rearrangement-v0")
class RearrangementDatasetV0(PointNavDatasetV1):
    r"""Class inherited from PointNavDataset that loads the Rearrangement dataset."""
    episodes: List[RearrangementEpisode]
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
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        for i, episode in enumerate(deserialized["episodes"]):
            episode_obj = RearrangementEpisode(**episode)
            episode_obj.episode_id = str(i)

            if scenes_dir is not None:
                if episode_obj.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode_obj.scene_id = episode_obj.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode_obj.scene_id = os.path.join(
                    scenes_dir, episode_obj.scene_id
                )

            for i, obj in enumerate(episode_obj.objects):
                idx = obj["object_key"]
                if type(idx) is not str:
                    template = episode_obj.object_templates[idx]
                    obj["object_key"] = template["object_key"]
                    obj["object_template"] = template["object_template"]
                episode_obj.objects[i] = RearrangementObjectSpec(**obj)

            for i, goal in enumerate(episode_obj.goals):
                episode_obj.goals[i] = RearrangementSpec(**goal)

            self.episodes.append(episode_obj)
