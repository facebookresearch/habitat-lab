from collections import OrderedDict
from enum import Enum
from typing import Any, Dict, List, Tuple

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

    def __init__(self) -> None:
        self.uuid: str
        self.sensor_type: SensorTypes
        self.observation_space: Space

    def get_observation(self, **kwargs: Any) -> Any:
        r"""Returns the current observation for Sensor.
        """
        raise NotImplementedError


class Observation(dict):
    r"""Represents an observation provided by a sensor.

    Thin wrapper of OrderedDict with potentially some utility functions
    to obtain Tensors)
    """

    def __init__(self, sensors: Dict[str, Sensor], **kwargs) -> None:
        data = [(uuid, sensor.get_observation(**kwargs)) for uuid, sensor in
                sensors.items()]
        super().__init__(data)


class RGBSensor(Sensor):
    def __init__(self, **kwargs):
        self.uuid = 'rgb'
        self.sensor_type = SensorTypes.COLOR

    def get_observation(self, **kwargs: Any) -> Any:
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
            assert sensor.uuid not in self.sensors, \
                "'{}' is duplicated sensor uuid".format(sensor.uuid)
            self.sensors[sensor.uuid] = sensor
            spaces[sensor.uuid] = sensor.observation_space
        self.observation_spaces: SpaceDict = SpaceDict(spaces=spaces)

    def get(self, uuid: str) -> Sensor:
        return self.sensors[uuid]

    def get_observations(self, **kwargs: Any) -> Observation:
        r"""
        :return: collect data from all sensors packaged into Observation
        """
        return Observation(self.sensors, **kwargs)


class Simulator:
    def reset(self) -> Observation:
        raise NotImplementedError

    def step(self, action: int) -> Tuple[Observation, bool]:
        raise NotImplementedError

    def seed(self, seed: int) -> None:
        raise NotImplementedError

    def reconfigure(self, config: Any) -> None:
        raise NotImplementedError

    def geodesic_distance(self, position_a: List[float],
                          position_b: List[float]) -> float:
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

    def close(self) -> None:
        raise NotImplementedError
