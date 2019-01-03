from typing import Any, Dict, List, Optional, Type

import numpy as np
from gym import spaces

import teas
from teas.core.dataset import Episode, Dataset
from teas.core.simulator import (
    Observation, SensorSuite, Sensor,
    SensorTypes, Simulator
)


class ShortestPathPoint:
    def __init__(self, position: List[Any], rotation: List[Any],
                 action: Optional[int]) -> None:
        self.position: List[Any] = position
        self.rotation: List[Any] = rotation
        self.action: Optional[int] = action


class NavigationGoal:
    r"""Base class for a goal specification hierarchy.
    """

    def __init__(self, position: List[float], radius: Optional[float] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)  # type: ignore
        self.position: List[float] = position
        self.radius: Optional[float] = radius


class ObjectGoal(NavigationGoal):
    r"""Object goal that can be specified by object_id or position or object
    category.
    """

    def __init__(self,
                 object_id: str,
                 room_id: Optional[str] = None,
                 object_name: Optional[str] = None,
                 object_category: Optional[str] = None,
                 room_name: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.object_id: str = object_id
        self.object_name: Optional[str] = object_name
        self.object_category: Optional[str] = object_category
        self.room_id: Optional[str] = room_id
        self.room_name: Optional[str] = room_name


class RoomGoal(NavigationGoal):
    r"""Room goal that can be specified by room_id or position with radius.
    """

    def __init__(self, room_id: str, room_name: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)  # type: ignore
        self.room_id: str = room_id
        self.room_name: Optional[str] = room_name


class NavigationEpisode(Episode):
    r"""Class for episode specification that includes initial position and
    rotation of agent, scene name, goal and optional shortest paths. An
    episode is a description of one task instance for the agent.
    """

    def __init__(self,
                 goals: List[NavigationGoal],
                 start_room: Optional[str] = None,
                 shortest_paths: Optional[List[ShortestPathPoint]] = None,
                 **kwargs) -> None:
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
        self.start_room: Optional[str] = start_room
        self.goals: List[NavigationGoal] = goals
        self.shortest_paths: Optional[List[ShortestPathPoint]] = shortest_paths


# TODO (maksymets) Move reward to measurement class
class RewardSensor(Sensor):
    REWARD_MIN = -100
    REWARD_MAX = -100

    def __init__(self, **kwargs):
        self.uuid = 'reward'
        self.sensor_type = SensorTypes.TENSOR
        self.observation_space = spaces.Box(low=RewardSensor.REWARD_MIN,
                                            high=RewardSensor.REWARD_MAX,
                                            shape=(1,),
                                            dtype=np.float)

    def _get_observation(self, observations: Dict[str, Observation],
                         episode: NavigationEpisode,
                         **kwargs):
        return [0]

    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)


class NavigationTask(teas.EmbodiedTask):
    REWARD_ID = "reward"
    DONE_ID = "done"

    def __init__(self, config: Any, simulator: Simulator,
                 dataset: Optional[Dataset] = None) -> None:
        self._config: Any = config
        self._simulator = simulator
        self._dataset: Optional[Dataset] = dataset
        self._sensor_suite: SensorSuite = SensorSuite([RewardSensor()])

    def get_reward(self, observations: Dict[str, Observation]) -> Any:
        return observations[NavigationTask.REWARD_ID]

    def overwrite_sim_config(self, sim_config: Any,
                             episode: Type[Episode]) -> Any:
        sim_config.scene = episode.scene_id
        if episode.start_position is not None and \
                episode.start_rotation is not None:
            # yacs config attributes cannot be None
            sim_config.start_position = episode.start_position
            sim_config.start_rotation = episode.start_rotation
        return sim_config
