from collections import OrderedDict
from enum import Enum

from gym.spaces.dict_space import Dict


class Observation(OrderedDict):
    r"""Represents an observation 'frame' provided by a sensor.

    Thin wrapper of OrderedDict with potentially some utility functions
    to obtain Tensors)
    """
    
    def __init__(self, sensors):
        data = [(uuid, sensor.observation()) for uuid, sensor in
                sensors.items()]
        super().__init__(data)


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
    r"""Represents a sensor that provides data from the environment to an agent.
    
    The user of this class needs to implement:
    
        observation
    
    The user of this class is required to set the following attributes:
    
        uuid: universally unique id.
        sensor_type: type of Sensor, use SensorTypes enum if your sensor
        comes under one of it's categories.
        observation_space: gym.Space object corresponding to observation of
        sensor
    """
    uuid = None
    sensor_type = None
    observation_space = None
    
    def observation(self):
        r"""Returns the current observation for Sensor.
        """
        raise NotImplementedError


class RGBSensor(Sensor):
    def __init__(self, *args):
        self.uuid = 'rgb'
        self.sensor_type = SensorTypes.COLOR
    
    def observation(self):
        raise NotImplementedError


class SensorSuite:
    r"""Represents a set of sensors, with each sensor being identified
    through a unique id.
    """
    
    def __init__(self, sensors):
        r"""
        Args
            sensors: list containing sensors for the environment, the uuid of
            each sensor should be unique.
        """
        self.sensors = OrderedDict()
        spaces = OrderedDict()
        for sensor in sensors:
            assert sensor.uuid not in self.sensors, "duplicate sensor uuid"
            self.sensors[sensor.uuid] = sensor
            spaces[sensor.uuid] = sensor.observation_space
        self.observation_spaces = Dict(spaces=spaces)
    
    def get(self, uuid):
        return self.sensors[uuid]
    
    def observations(self) -> Observation:
        r"""
        :return: collect data from all sensors packaged into Observation
        """
        return Observation(self.sensors)


class Simulator:
    def reset(self):
        raise NotImplementedError
    
    def step(self, action):
        raise NotImplementedError
    
    def seed(self, seed):
        raise NotImplementedError
    
    def reconfigure(self, *config):
        raise NotImplementedError
    
    def close(self):
        raise NotImplementedError
