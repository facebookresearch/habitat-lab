from collections import OrderedDict
from enum import Enum
from typing import Any, Dict, List, Tuple, Optional

from gym import Space
from gym.spaces.dict_space import Dict as SpaceDict


class SensorTypes(Enum):
    r"""Enumeration of types of sensors.
    """
    NULL = 0
    COLOR = 1
    DEPTH = 2
    NORMAL = 3
    SEMANTIC = 4
    PATH = 5
    GOAL = 6
    FORCE = 7
    TENSOR = 8
    TEXT = 9


class Sensor:
    r"""Represents a sensor that provides data from the environment to agent.

    The user of this class needs to implement:

        observation

    The user of this class is required to set the following attributes:

        uuid: universally unique id.
        sensor_type: type of Sensor, use SensorTypes enum if your sensor
        comes under one of it's categories.
        observation_space: gym.Space object corresponding to observation of
        sensor
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.uuid: str = self._get_uuid(*args, **kwargs)
        self.sensor_type: SensorTypes = self._get_sensor_type(*args, **kwargs)
        self.observation_space: Space = self._get_observation_space(
            *args, **kwargs
        )

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        raise NotImplementedError

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        raise NotImplementedError

    def get_observation(self, *args: Any, **kwargs: Any) -> Any:
        r"""Returns the current observation for Sensor.
        """
        raise NotImplementedError


class Observations(dict):
    r"""Dict containing sensor observations

    Thin wrapper of OrderedDict with potentially some utility functions
    to obtain Tensors)
    """

    def __init__(self, sensors: Dict[str, Sensor], *args, **kwargs) -> None:
        data = [
            (uuid, sensor.get_observation(*args, **kwargs))
            for uuid, sensor in sensors.items()
        ]
        super().__init__(data)


class RGBSensor(Sensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args, **kwargs):
        return "rgb"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args, **kwargs):
        raise NotImplementedError

    def get_observation(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class DepthSensor(Sensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args, **kwargs):
        return "depth"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.DEPTH

    def _get_observation_space(self, *args, **kwargs):
        raise NotImplementedError

    def get_observation(self, *args: Any, **kwargs: Any):
        raise NotImplementedError


class SemanticSensor(Sensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args, **kwargs):
        return "semantic"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args, **kwargs):
        raise NotImplementedError

    def get_observation(self, *args: Any, **kwargs: Any):
        raise NotImplementedError


class SensorSuite:
    r"""Represents a set of sensors, with each sensor being identified
    through a unique id.
    """

    def __init__(self, sensors: List[Sensor]) -> None:
        r"""
        Args
            sensors: list containing sensors for the environment, the uuid of
            each sensor should be unique.
        """
        self.sensors: OrderedDict[str, Sensor] = OrderedDict()
        spaces: OrderedDict[str, Space] = OrderedDict()
        for sensor in sensors:
            assert (
                sensor.uuid not in self.sensors
            ), "'{}' is duplicated sensor uuid".format(sensor.uuid)
            self.sensors[sensor.uuid] = sensor
            spaces[sensor.uuid] = sensor.observation_space
        self.observation_spaces: SpaceDict = SpaceDict(spaces=spaces)

    def get(self, uuid: str) -> Sensor:
        return self.sensors[uuid]

    def get_observations(self, *args: Any, **kwargs: Any) -> Observations:
        r"""
        :return: collect data from all sensors packaged into Observation
        """
        return Observations(self.sensors, *args, **kwargs)


class AgentState:
    position: List[float]
    rotation: Optional[List[float]]

    def __init__(
        self, position: List[float], rotation: Optional[List[float]]
    ) -> None:
        self.position = position
        self.rotation = rotation


class ShortestPathPoint:
    position: List[Any]
    rotation: List[Any]
    action: Optional[int]

    def __init__(
        self, position: List[Any], rotation: List[Any], action: Optional[int]
    ) -> None:
        self.position = position
        self.rotation = rotation
        self.action = action


class Simulator:
    def reset(self) -> Observations:
        raise NotImplementedError

    def step(self, action: int) -> Observations:
        raise NotImplementedError

    def seed(self, seed: int) -> None:
        raise NotImplementedError

    def reconfigure(self, config: Any) -> None:
        raise NotImplementedError

    def geodesic_distance(
        self, position_a: List[float], position_b: List[float]
    ) -> float:
        r"""
        :param position_a: starting point for distance calculation
        :param position_b: ending point for distance calculation
        :return: the geodesic distance in the cartesian space between points
                 position_a and position_b, if no path is found between the
                 points then infinity is returned.
        """
        raise NotImplementedError

    def sample_navigable_point(self) -> List[float]:
        r"""
        :return: a random navigable point from the simulator. A point is
        defined as navigable if the agent can be initialized at the point.
        """
        raise NotImplementedError

    def action_space_shortest_paths(
        self, source: AgentState, targets: List[AgentState], agent_id: int
    ) -> List[ShortestPathPoint]:
        r"""
        :param source: source agent state for shortest path calculation
        :param targets: target agent state(s) for shortest path calculation
        :param agent_id: int identification of agent from multi-agent setup
        :return: List of agent states and actions along the shortest path from
        source to the nearest target (both included). If one of the target(s)
        is identical to the source, a list containing only one node with the
        identical agent state is returned. Returns an empty list in case none
        of the targets are reachable from the source. For the last item in
        the returned list the action will be None
        """
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
