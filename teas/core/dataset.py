import json
from typing import List, Type, TypeVar, Generic


class Episode:
    r"""Base class for episode specification that includes initial position and
    rotation of agent, scene id, episode id provided by dataset. An
    episode is a description of one task instance for the agent.
    """
    episode_id: str
    scene_id: str
    start_position: List[float]
    start_rotation: List[float]

    def __init__(self, episode_id: str, scene_id: str,
                 start_position: List[float],
                 start_rotation: List[float]) -> None:
        r"""
        :param episode_id: id of episode in the dataset, usually episode number
        :param scene_id: id of scene in scene dataset
        :param start_position: numpy ndarray containing 3 entries for (x, y, z)
        :param start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
        elements of unit quaternion (versor) representing agent 3D orientation,
        ref: https://en.wikipedia.org/wiki/Versor
        """
        self.episode_id = episode_id
        self.scene_id = scene_id
        self.start_position = start_position
        self.start_rotation = start_rotation

    def __str__(self):
        return str(self.__dict__)


T = TypeVar('T', Episode, Type[Episode])


class Dataset(Generic[T]):
    r"""Base class for dataset specification that includes list of
    episode and relevant method to access episodes from particular
    scene as well as scene id list.
    """
    episodes: List[T]

    @property
    def scene_ids(self) -> List[str]:
        r"""Return list of scene ids for which dataset has episodes.
        """
        return list({episode.scene_id for episode in self.episodes})

    def get_scene_episodes(self, scene_id: str) -> List[T]:
        r"""Return list of episodes for particular scene_id.
        :param scene_id: id of scene in scene dataset
        """
        return list(filter(lambda x: x.scene_id == scene_id,
                           iter(self.episodes)))

    def get_episodes(self, indexes: List[int]) -> List[T]:
        r"""Return list of episodes with particular episode indexes.
        :param indexes: indexes of episodes in dataset
        """
        return [self.episodes[episode_id] for episode_id in indexes]

    def to_json(self) -> str:
        class DatasetJSONEncoder(json.JSONEncoder):
            def default(self, object):
                return object.__dict__

        # TODO(maksymets): remove call of internal DatasetFloatJSONEncoder
        #  used for float precision decrease
        from teas.internal.data.datasets.utils import DatasetFloatJSONEncoder
        result = DatasetFloatJSONEncoder().encode(self)
        return result

    def from_json(self, serialized: str) -> None:
        raise NotImplementedError
