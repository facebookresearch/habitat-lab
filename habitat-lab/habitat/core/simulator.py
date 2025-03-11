#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Defines the core Simulator and Sensor class wrapper APIs. The classes here are primarily defining abstract APIs which are implemented further downstream."""
import abc
import time
from collections import OrderedDict
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Union,
)

import attr
import numpy as np
import quaternion
from gym import Space, spaces

from habitat.core.dataset import Episode

if TYPE_CHECKING:
    try:
        from torch import Tensor
    except ImportError:
        pass
    from omegaconf import DictConfig

VisualObservation = Union[np.ndarray, "Tensor"]


@attr.s(auto_attribs=True)
class ActionSpaceConfiguration(metaclass=abc.ABCMeta):
    """Attrs base class wrapper for DictConfig defining the action space for a task.

    :property config: The action space DictConfig.
    """

    config: "DictConfig"

    @abc.abstractmethod
    def get(self) -> Any:
        raise NotImplementedError


class SensorTypes(Enum):
    """Enumeration of types of sensors."""

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
    TOKEN_IDS = 13


class Sensor(metaclass=abc.ABCMeta):
    """Represents a sensor that provides data from the environment to agent. The user of this class needs to implement the get_observation method and the user is also required to set attributes.

    :data uuid: universally unique id.
    :data sensor_type: type of Sensor, use SensorTypes enum if your sensor comes under one of it's categories.
    :data observation_space: ``gym.Space`` object corresponding to observation of sensor.
    :data config: The SensorConfig.
    """

    uuid: str
    config: "DictConfig"
    sensor_type: SensorTypes
    observation_space: Space

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.config = kwargs["config"] if "config" in kwargs else None
        if hasattr(self.config, "uuid"):
            # We allow any sensor config to override the uuid
            self.uuid = self.config.uuid
        else:
            self.uuid = self._get_uuid(*args, **kwargs)
        self.sensor_type = self._get_sensor_type(*args, **kwargs)
        self.observation_space = self._get_observation_space(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        raise NotImplementedError

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        raise NotImplementedError

    @abc.abstractmethod
    def get_observation(self, *args: Any, **kwargs: Any) -> Any:
        """Returns current observation for Sensor."""
        raise NotImplementedError


class Observations(Dict[str, Any]):
    """Dictionary containing sensor observations"""

    def __init__(
        self,
        sensors: Dict[str, Sensor],
        *args: Any,
        should_time: bool = False,
        **kwargs: Any,
    ) -> None:
        """..

        :param sensors: list of sensors whose observations are fetched and packaged.
        :param should_time: Optionally log performance timing metrics.
        """
        data = []
        for uuid, sensor in sensors.items():
            t_start = time.time()
            data.append((uuid, sensor.get_observation(*args, **kwargs)))

            if should_time:
                kwargs["task"].add_perf_timing(f"sensors.{uuid}", t_start)

        super().__init__(data)


class RGBSensor(Sensor, metaclass=abc.ABCMeta):
    """Wrapper for 3-channel color Camera Sensors. See :ref:`Sensor` parent class for more details."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "rgb"

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        raise NotImplementedError

    def get_observation(self, *args: Any, **kwargs: Any) -> VisualObservation:
        raise NotImplementedError


class DepthSensor(Sensor, metaclass=abc.ABCMeta):
    """Wrapper for depth Camera Sensors. See :ref:`Sensor` parent class for more details."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "depth"

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.DEPTH

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        raise NotImplementedError

    def get_observation(self, *args: Any, **kwargs: Any) -> VisualObservation:
        raise NotImplementedError


class SemanticSensor(Sensor):
    """Wrapper for integer id Camera Sensors where each integer is mapped to an object instance or semantic class. See :ref:`Sensor` parent class for more details."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "semantic"

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        raise NotImplementedError

    def get_observation(self, *args: Any, **kwargs: Any) -> VisualObservation:
        raise NotImplementedError


class BumpSensor(Sensor):
    """Wrapper for non-visual navmesh collision Sensors. See :ref:`Sensor` parent class for more details."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "bump"

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.FORCE

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        raise NotImplementedError

    def get_observation(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class SensorSuite:
    """Represents a set of sensors, with each sensor being identified
    through a unique id.

    :data sensors: dict mapping Sensors' uids to their Sensor objects.
    :data observation_spaces: dict of observation spaces for each sensor keyed by uid.
    """

    sensors: Dict[str, Sensor]
    observation_spaces: spaces.Dict

    def __init__(self, sensors: Iterable[Sensor]) -> None:
        """..

        :param sensors: list containing sensors for the environment, uuid of
            each sensor must be unique.
        """
        self.sensors = OrderedDict()
        ordered_spaces: OrderedDict[str, Space] = OrderedDict()
        for sensor in sensors:
            assert (
                sensor.uuid not in self.sensors
            ), "'{}' is duplicated sensor uuid".format(sensor.uuid)
            self.sensors[sensor.uuid] = sensor
            ordered_spaces[sensor.uuid] = sensor.observation_space
        self.observation_spaces = spaces.Dict(spaces=ordered_spaces)

    def get(self, uuid: str) -> Sensor:
        return self.sensors[uuid]

    def get_observations(self, *args: Any, **kwargs: Any) -> Observations:
        """Collects data from all sensors and returns it packaged inside
        :ref:`Observations`.
        """
        return Observations(self.sensors, *args, **kwargs)


@attr.s(auto_attribs=True)
class AgentState:
    """Represents the rigid transformation state of an agent as a 3D position and quaternion rotation.

    :property position: 3D position of the agent's base. Typically on the navmesh.
    :property rotation: quaternion orientation of the agent's base. Typically a yaw rotation.
    """

    position: Union[None, List[float], np.ndarray]
    rotation: Union[None, np.ndarray, quaternion.quaternion] = None


@attr.s(auto_attribs=True)
class ShortestPathPoint:
    """Wrapper for the information embedded in a single point for a ShortestPath planner object: 3D position, quaternion rotation, and the action which led to the state.

    :property position: 3D global position of the path point. Typically corresponds to an agent's base position on the navmesh.
    :property rotation: quaternion orientation of the agent at the point.
    :property action: the action, typically a discrete transformation, which led to the this path point for use in path planning.
    """

    position: List[Any]
    rotation: List[Any]
    action: Union[int, np.ndarray, None] = None


class Simulator:
    """Abstract simulator class for habitat. New simulators to be added to habitat
    must derive from this class and implement the abstract methods.

    :data habitat_config: The Dictconfig object containing configuration parameters specifically pertaining to the habitat Simulator.
    """

    habitat_config: "DictConfig"

    def __init__(self, *args, **kwargs) -> None:
        pass

    @property
    def sensor_suite(self) -> SensorSuite:
        raise NotImplementedError

    @property
    def action_space(self) -> Space:
        raise NotImplementedError

    def reset(self) -> Observations:
        """resets the simulator and returns the initial observations.

        :return: initial observations from simulator.
        """
        raise NotImplementedError

    def step(self, action, *args, **kwargs) -> Observations:
        """Perform an action in the simulator and return observations.

        :param action: action to be performed inside the simulator.
        :return: observations after taking action in simulator.
        """
        raise NotImplementedError

    def seed(self, seed: int) -> None:
        raise NotImplementedError

    def reconfigure(
        self, config: "DictConfig", episode: Optional[Episode] = None
    ) -> None:
        raise NotImplementedError

    def geodesic_distance(
        self,
        position_a: Union[Sequence[float], np.ndarray],
        position_b: Union[
            Sequence[float], Sequence[Sequence[float]], np.ndarray
        ],
        episode: Optional[Episode] = None,
    ) -> float:
        """Calculates geodesic distance between two points.

        :param position_a: coordinates of first point.
        :param position_b: coordinates of second point or list of goal points
            coordinates.
        :param episode: The episode with these ends points.  This is used for
            shortest path computation caching
        :return:
            the geodesic distance in the cartesian space between points
            :p:`position_a` and :p:`position_b`, if no path is found between
            the points then :ref:`math.inf` is returned.
        """
        raise NotImplementedError

    def get_agent_state(self, agent_id: int = 0) -> AgentState:
        """..

        :param agent_id: id of agent.
        :return: state of agent corresponding to :p:`agent_id`.
        """
        raise NotImplementedError

    def get_observations_at(
        self,
        position: List[float],
        rotation: List[float],
        keep_agent_at_new_pose: bool = False,
    ) -> Optional[Observations]:
        """Returns the observation.

        :param position: list containing 3 entries for :py:`(x, y, z)`.
        :param rotation: list with 4 entries for :py:`(x, y, z, w)` elements
            of unit quaternion (versor) representing agent 3D orientation,
            (https://en.wikipedia.org/wiki/Versor)
        :param keep_agent_at_new_pose: If true, the agent will stay at the
            requested location. Otherwise it will return to where it started.
        :return:
            The observations or :py:`None` if it was unable to get valid
            observations.

        """
        raise NotImplementedError

    def sample_navigable_point(self) -> List[float]:
        """Samples a navigable point from the simulator. A point is defined as
        navigable if the agent can be initialized at that point.

        :return: navigable point.
        """
        raise NotImplementedError

    def is_navigable(self, point: List[float]) -> bool:
        """Return :py:`True` if the agent can stand at the specified point.

        :param point: the point to check.
        """
        raise NotImplementedError

    def action_space_shortest_path(
        self, source: AgentState, targets: List[AgentState], agent_id: int = 0
    ) -> List[ShortestPathPoint]:
        """Calculates the shortest path between source and target agent
        states.

        :param source: source agent state for shortest path calculation.
        :param targets: target agent state(s) for shortest path calculation.
        :param agent_id: id for agent (relevant for multi-agent setup).
        :return: list of agent states and actions along the shortest path from
            source to the nearest target (both included).
        """
        raise NotImplementedError

    def get_straight_shortest_path_points(
        self, position_a: List[float], position_b: List[float]
    ) -> List[List[float]]:
        """Returns points along the geodesic (shortest) path between two
        points irrespective of the angles between the waypoints.

        :param position_a: the start point. This will be the first point in
            the returned list.
        :param position_b: the end point. This will be the last point in the
            returned list.
        :return: a list of waypoints :py:`(x, y, z)` on the geodesic path
            between the two points.
        """

        raise NotImplementedError

    @property
    def up_vector(self) -> "np.ndarray":
        """The vector representing the direction upward (perpendicular to the
        floor) from the global coordinate frame.
        """
        raise NotImplementedError

    @property
    def forward_vector(self) -> "np.ndarray":
        """The forward direction in the global coordinate frame i.e. the
        direction of forward movement for an agent with 0 degrees rotation in
        the ground plane.
        """
        raise NotImplementedError

    def render(self, mode: str = "rgb") -> Any:
        raise NotImplementedError

    def close(self, destroy: bool = True) -> None:
        pass

    def previous_step_collided(self) -> bool:
        """Whether or not the previous step resulted in a collision

        :return: :py:`True` if the previous step resulted in a collision,
            :py:`False` otherwise
        """
        raise NotImplementedError

    def __enter__(self) -> "Simulator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
