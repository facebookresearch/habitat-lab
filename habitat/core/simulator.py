#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from collections import OrderedDict
from enum import Enum
from typing import Any, Dict, List, Optional

import attr
from gym import Space
from gym.spaces.dict_space import Dict as SpaceDict

from habitat.config import Config
from habitat.core.utils import Singleton


@attr.s(auto_attribs=True)
class ActionSpaceConfiguration(abc.ABC):
    config: Config

    @abc.abstractmethod
    def get(self):
        pass


class _DefaultSimulatorActions(Enum):
    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    LOOK_UP = 4
    LOOK_DOWN = 5


@attr.s(auto_attribs=True, slots=True)
class SimulatorActionsSingleton(metaclass=Singleton):
    r"""Implements an extendable Enum for the mapping of action names
    to their integer values.

    This means that new action names can be added, but old action names cannot
    be removed nor can their mapping be altered. This also ensures that all
    actions are always contigously mapped in :py:`[0, len(SimulatorActions) - 1]`

    This accesible as the global singleton `SimulatorActions`
    """

    _known_actions: Dict[str, int] = attr.ib(init=False, factory=dict)

    def __attrs_post_init__(self):
        for action in _DefaultSimulatorActions:
            self._known_actions[action.name] = action.value

    def extend_action_space(self, name: str) -> int:
        r"""Extends the action space to accomidate a new action with
        the name :p:`name`

        :param name: The name of the new action
        :return: The number the action is registered on

        Usage:

        .. code:: py

            from habitat import SimulatorActions
            SimulatorActions.extend_action_space("MY_ACTION")
            print(SimulatorActions.MY_ACTION)
        """
        assert (
            name not in self._known_actions
        ), "Cannot register an action name twice"
        self._known_actions[name] = len(self._known_actions)

        return self._known_actions[name]

    def has_action(self, name: str) -> bool:
        r"""Checks to see if action :p:`name` is already register

        :param name: The name to check
        :return: Whether or not :p:`name` already exists
        """

        return name in self._known_actions

    def __getattr__(self, name):
        return self._known_actions[name]

    def __getitem__(self, name):
        return self._known_actions[name]

    def __len__(self):
        return len(self._known_actions)

    def __iter__(self):
        return iter(self._known_actions)


SimulatorActions: SimulatorActionsSingleton = SimulatorActionsSingleton()


class SensorTypes(Enum):
    r"""Enumeration of types of sensors.
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
    r"""Represents a sensor that provides data from the environment to agent.

    :data uuid: universally unique id.
    :data sensor_type: type of Sensor, use SensorTypes enum if your sensor
        comes under one of it's categories.
    :data observation_space: ``gym.Space`` object corresponding to observation
        of sensor.

    The user of this class needs to implement the get_observation method and
    the user is also required to set the below attributes:
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
        r"""
        Returns:
            current observation for Sensor.
        """
        raise NotImplementedError


class Observations(dict):
    r"""Dictionary containing sensor observations
    """

    def __init__(
        self, sensors: Dict[str, Sensor], *args: Any, **kwargs: Any
    ) -> None:
        """Constructor

        :param sensors: list of sensors whose observations are fetched and
            packaged.
        """

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
    r"""Represents a set of sensors, with each sensor being identified
    through a unique id.
    """

    sensors: Dict[str, Sensor]
    observation_spaces: SpaceDict

    def __init__(self, sensors: List[Sensor]) -> None:
        """Constructor

        :param sensors: list containing sensors for the environment, uuid of
            each sensor must be unique.
        """
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
        r"""Collects data from all sensors and returns it packaged inside
            `Observations`.
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
    r"""Basic simulator class for habitat. New simulators to be added to habtiat
    must derive from this class and implement the abstarct methods.
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
        r"""resets the simulator and returns the initial observations.

        :return: initial observations from simulator.
        """
        raise NotImplementedError

    def step(self, action: int) -> Observations:
        r"""Perform an action in the simulator and return observations.

        :param action: action to be performed inside the simulator.
        :return: observations after taking action in simulator.
        """
        raise NotImplementedError

    def seed(self, seed: int) -> None:
        raise NotImplementedError

    def reconfigure(self, config: Config) -> None:
        raise NotImplementedError

    def geodesic_distance(
        self, position_a: List[float], position_b: List[float]
    ) -> float:
        r"""Calculates geodesic distance between two points.

        :param position_a: coordinates of first point.
        :param position_b: coordinates of second point.
        :return:
            the geodesic distance in the cartesian space between points
            :p:`position_a` and :p:`position_b`, if no path is found between
            the points then `math.inf` is returned.
        """
        raise NotImplementedError

    def get_agent_state(self, agent_id: int = 0):
        r"""..

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
        r"""Samples a navigable point from the simulator. A point is defined as
        navigable if the agent can be initialized at that point.

        :return: navigable point.
        """
        raise NotImplementedError

    def is_navigable(self, point: List[float]) -> bool:
        r"""Return :py:`True` if the agent can stand at the specified point.

        :param point: the point to check.
        """
        raise NotImplementedError

    def action_space_shortest_path(
        self, source: AgentState, targets: List[AgentState], agent_id: int = 0
    ) -> List[ShortestPathPoint]:
        r"""Calculates the shortest path between source and target agent
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
        r"""Returns points along the geodesic (shortest) path between two
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
    def up_vector(self):
        r"""The vector representing the direction upward (perpendicular to the
        floor) from the global coordinate frame.
        """
        raise NotImplementedError

    @property
    def forward_vector(self):
        r"""The forward direction in the global coordinate frame i.e. the
        direction of forward movement for an agent with 0 degrees rotation in
        the ground plane.
        """
        raise NotImplementedError

    def render(self, mode: str = "rgb") -> Any:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    @property
    def index_stop_action(self):
        return SimulatorActions.STOP

    @property
    def index_forward_action(self):
        return SimulatorActions.MOVE_FORWARD

    def previous_step_collided(self) -> bool:
        r"""Whether or not the previous step resulted in a collision

        :return: :py:`True` if the previous step resulted in a collision,
            :py:`False` otherwise
        """
        raise NotImplementedError
