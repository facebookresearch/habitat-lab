#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from typing import Any, Dict, List, Optional
from enum import Enum

from gym import Space
from gym.spaces.dict_space import Dict as SpaceDict
from habitat.config import Config


class SensorTypes(Enum):
    """Enumeration of types of sensors.
    """

    NULL = 0
    COLOR = 1
    DEPTH = 2
    NORMAL = 3
    SEMANTIC = 4
    PATH = 5
    POSITION = 6
    FORCE = 7
    TENSOR = 8
    TEXT = 9
    MEASUREMENT = 10
    HEADING = 11
    TACTILE = 12


class Sensor:
    """Represents a sensor that provides data from the environment to agent.
    The user of this class needs to implement the get_observation method and
    the user is also required to set the below attributes:

    Attributes:
        uuid: universally unique id.
        sensor_type: type of Sensor, use SensorTypes enum if your sensor
            comes under one of it's categories.
        observation_space: gym.Space object corresponding to observation of
            sensor
    """

    uuid: str
    config: Config
    sensor_type: SensorTypes
    observation_space: Space

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.config = kwargs["config"] if "config" in kwargs else None
        self.uuid = self._get_uuid(*args, **kwargs)
        self.sensor_type = self._get_sensor_type(*args, **kwargs)
        self.observation_space = self._get_observation_space(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        raise NotImplementedError

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        raise NotImplementedError

    def get_observation(self, *args: Any, **kwargs: Any) -> Any:
        """
        Returns:
            Current observation for Sensor.
        """
        raise NotImplementedError


class Observations(dict):
    """Dictionary containing sensor observations

    Args:
        sensors: list of sensors whose observations are fetched and packaged.
    """

    def __init__(
        self, sensors: Dict[str, Sensor], *args: Any, **kwargs: Any
    ) -> None:
        data = [
            (uuid, sensor.get_observation(*args, **kwargs))
            for uuid, sensor in sensors.items()
        ]
        super().__init__(data)


class RGBSensor(Sensor):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "rgb"

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        raise NotImplementedError

    def get_observation(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class DepthSensor(Sensor):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "depth"

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.DEPTH

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        raise NotImplementedError

    def get_observation(self, *args: Any, **kwargs: Any):
        raise NotImplementedError


class SemanticSensor(Sensor):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "semantic"

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        raise NotImplementedError

    def get_observation(self, *args: Any, **kwargs: Any):
        raise NotImplementedError


class SensorSuite:
    """Represents a set of sensors, with each sensor being identified
    through a unique id.

    Args:
        sensors: list containing sensors for the environment, uuid of each
            sensor must be unique.
    """

    sensors: Dict[str, Sensor]
    observation_spaces: SpaceDict

    def __init__(self, sensors: List[Sensor]) -> None:
        self.sensors = OrderedDict()
        spaces: OrderedDict[str, Space] = OrderedDict()
        for sensor in sensors:
            assert (
                sensor.uuid not in self.sensors
            ), "'{}' is duplicated sensor uuid".format(sensor.uuid)
            self.sensors[sensor.uuid] = sensor
            spaces[sensor.uuid] = sensor.observation_space
        self.observation_spaces = SpaceDict(spaces=spaces)

    def get(self, uuid: str) -> Sensor:
        return self.sensors[uuid]

    def get_observations(self, *args: Any, **kwargs: Any) -> Observations:
        """
        Returns:
            collect data from all sensors and return it packaged inside
            Observation.
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
    """Basic simulator class for habitat. New simulators to be added to habtiat
    must derive from this class and implement the below methods:
        reset
        step
        seed
        reconfigure
        geodesic_distance
        sample_navigable_point
        action_space_shortest_path
        close
    """

    @property
    def sensor_suite(self) -> SensorSuite:
        raise NotImplementedError

    @property
    def action_space(self) -> Space:
        raise NotImplementedError

    @property
    def is_episode_active(self) -> bool:
        raise NotImplementedError

    def reset(self) -> Observations:
        """Resets the simulator and returns the initial observations.

        Returns:
            Initial observations from simulator.
        """
        raise NotImplementedError

    def step(self, action: int) -> Observations:
        """Perform an action in the simulator and return observations.

        Args:
            action: action to be performed inside the simulator.

        Returns:
            observations after taking action in simulator.
        """
        raise NotImplementedError

    def seed(self, seed: int) -> None:
        raise NotImplementedError

    def reconfigure(self, config: Config) -> None:
        raise NotImplementedError

    def geodesic_distance(
        self, position_a: List[float], position_b: List[float]
    ) -> float:
        """Calculates geodesic distance between two points.

        Args:
            position_a: coordinates of first point
            position_b: coordinates of second point

        Returns:
            the geodesic distance in the cartesian space between points
            position_a and position_b, if no path is found between the
            points then infinity is returned.
        """
        raise NotImplementedError

    def get_agent_state(self, agent_id: int = 0):
        """
        Args:
             agent_id: id of agent

        Returns:
            state of agent corresponding to agent_id
        """
        raise NotImplementedError

    def sample_navigable_point(self) -> List[float]:
        """Samples a navigable point from the simulator. A point is defined as
        navigable if the agent can be initialized at that point.

        Returns:
            Navigable point.
        """
        raise NotImplementedError

    def is_navigable(self, point: List[float]) -> bool:
        """Return true if the agent can stand at the specified point.

        Args:
            point: The point to check.
        """
        raise NotImplementedError

    def action_space_shortest_path(
        self, source: AgentState, targets: List[AgentState], agent_id: int = 0
    ) -> List[ShortestPathPoint]:
        """Calculates the shortest path between source and target agent states.

        Args:
            source: source agent state for shortest path calculation.
            targets: target agent state(s) for shortest path calculation.
            agent_id: id for agent (relevant for multi-agent setup).

        Returns:
            List of agent states and actions along the shortest path from
            source to the nearest target (both included).
        """
        raise NotImplementedError

    def get_straight_shortest_path_points(
        self, position_a: List[float], position_b: List[float]
    ) -> List[List[float]]:
        """Returns points along the geodesic (shortest) path between two points
         irrespective of the angles between the waypoints.

         Args:
            position_a: The start point. This will be the first point in the
                returned list.
            position_b: The end point. This will be the last point in the
                returned list.
        Returns:
            A list of waypoints (x, y, z) on the geodesic path between the two
            points.
         """

        raise NotImplementedError

    @property
    def up_vector(self):
        """The vector representing the direction upward (perpendicular to the
        floor) from the global coordinate frame.
        """
        raise NotImplementedError

    @property
    def forward_vector(self):
        """The forward direction in the global coordinate frame i.e. the
        direction of forward movement for an agent with 0 degrees rotation in
        the ground plane.
        """
        raise NotImplementedError

    def render(self, mode: str = "rgb") -> Any:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
