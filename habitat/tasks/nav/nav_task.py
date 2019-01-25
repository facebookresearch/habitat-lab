from typing import Any, Dict, List, Optional, Type

import habitat
from habitat.core.dataset import Episode, Dataset
from habitat.core.simulator import Observations, Simulator, ShortestPathPoint


def merge_sim_episode_config(sim_config: Any, episode: Type[Episode]) -> Any:
    sim_config.scene = episode.scene_id
    if (
        episode.start_position is not None
        and episode.start_rotation is not None
    ):
        # yacs config attributes cannot be None
        sim_config.start_position = episode.start_position
        sim_config.start_rotation = episode.start_rotation
    return sim_config


class NavigationGoal:
    r"""Base class for a goal specification hierarchy.
    """
    position: List[float]
    radius: Optional[float]

    def __init__(
        self, position: List[float], radius: Optional[float] = None, **kwargs
    ) -> None:
        self.position = position
        self.radius = radius


class ObjectGoal(NavigationGoal):
    r"""Object goal that can be specified by object_id or position or object
    category.
    """
    object_id: str
    object_name: Optional[str]
    object_category: Optional[str]
    room_id: Optional[str]
    room_name: Optional[str]

    def __init__(
        self,
        object_id: str,
        room_id: Optional[str] = None,
        object_name: Optional[str] = None,
        object_category: Optional[str] = None,
        room_name: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.object_id = object_id
        self.object_name = object_name
        self.object_category = object_category
        self.room_id = room_id
        self.room_name = room_name


class RoomGoal(NavigationGoal):
    r"""Room goal that can be specified by room_id or position with radius.
    """
    room_id: str
    room_name: Optional[str]

    def __init__(
        self, room_id: str, room_name: Optional[str] = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)  # type: ignore
        self.room_id = room_id
        self.room_name = room_name


class NavigationEpisode(Episode):
    r"""Class for episode specification that includes initial position and
    rotation of agent, scene name, goal and optional shortest paths. An
    episode is a description of one task instance for the agent.
    """
    goals: List[NavigationGoal]
    start_room: Optional[str]
    shortest_paths: Optional[List[ShortestPathPoint]]

    def __init__(
        self,
        goals: List[NavigationGoal],
        start_room: Optional[str] = None,
        shortest_paths: Optional[List[ShortestPathPoint]] = None,
        **kwargs
    ) -> None:
        r"""
        :param episode_id: id of episode in the dataset, usually episode number
        :param scene_id: id of scene in scene dataset
        :param start_position: numpy ndarray containing 3 entries for (x, y, z)
        :param start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
        elements of unit quaternion (versor) representing agent 3D orientation,
        ref: https://en.wikipedia.org/wiki/Versor
        :param goals: list of goals specifications
        :param: start_room: room id
        :param: shortest_paths: list containing shortest paths to goals
        """
        super().__init__(**kwargs)
        self.goals = goals
        self.shortest_paths = shortest_paths
        self.start_room = start_room


class NavigationTask(habitat.EmbodiedTask):
    def __init__(
        self, config: Any, sim: Simulator, dataset: Optional[Dataset] = None
    ) -> None:
        super().__init__(config=config, sim=sim, dataset=dataset)

    def overwrite_sim_config(
        self, sim_config: Any, episode: Type[Episode]
    ) -> Any:
        return merge_sim_episode_config(sim_config, episode)
