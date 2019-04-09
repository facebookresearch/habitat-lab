#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional, Type

import habitat
import numpy as np
from gym import spaces
from habitat.config import Config
from habitat.core.dataset import Episode, Dataset
from habitat.core.embodied_task import Measurements
from habitat.core.simulator import (
    Simulator,
    ShortestPathPoint,
    SensorTypes,
    SensorSuite,
)
from habitat.tasks.utils import quaternion_to_rotation, cartesian_to_polar

COLLISION_PROXIMITY_TOLERANCE: float = 1e-3


def merge_sim_episode_config(
    sim_config: Config, episode: Type[Episode]
) -> Any:
    sim_config.defrost()
    sim_config.SCENE = episode.scene_id
    sim_config.freeze()
    if (
        episode.start_position is not None
        and episode.start_rotation is not None
    ):
        agent_name = sim_config.AGENTS[sim_config.DEFAULT_AGENT_ID]
        agent_cfg = getattr(sim_config, agent_name)
        agent_cfg.defrost()
        agent_cfg.START_POSITION = episode.start_position
        agent_cfg.START_ROTATION = episode.start_rotation
        agent_cfg.IS_SET_START_STATE = True
        agent_cfg.freeze()
    return sim_config


class NavigationGoal:
    """Base class for a goal specification hierarchy.
    """

    position: List[float]
    radius: Optional[float]

    def __init__(
        self, position: List[float], radius: Optional[float] = None, **kwargs
    ) -> None:
        self.position = position
        self.radius = radius


class ObjectGoal(NavigationGoal):
    """Object goal that can be specified by object_id or position or object
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
    """Room goal that can be specified by room_id or position with radius.
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
    """Class for episode specification that includes initial position and
    rotation of agent, scene name, goal and optional shortest paths. An
    episode is a description of one task instance for the agent.

    Args:
        episode_id: id of episode in the dataset, usually episode number
        scene_id: id of scene in scene dataset
        start_position: numpy ndarray containing 3 entries for (x, y, z)
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation. ref: https://en.wikipedia.org/wiki/Versor
        goals: list of goals specifications
        start_room: room id
        shortest_paths: list containing shortest paths to goals
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
        super().__init__(**kwargs)
        self.goals = goals
        self.shortest_paths = shortest_paths
        self.start_room = start_room


class PointGoalSensor(habitat.Sensor):
    """
    Sensor for PointGoal observations which are used in the PointNav task.
    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PointGoal sensor. Can contain field for
            GOAL_FORMAT which can be used to specify the format in which
            the pointgoal is specified. Current options for goal format are
            cartesian and polar.

    Attributes:
        _goal_format: format for specifying the goal which can be done
            in cartesian or polar coordinates.
    """

    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim

        self._goal_format = getattr(config, "GOAL_FORMAT", "CARTESIAN")
        assert self._goal_format in ["CARTESIAN", "POLAR"]

        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "pointgoal"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        if self._goal_format == "CARTESIAN":
            sensor_shape = (3,)
        else:
            sensor_shape = (2,)
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode):
        agent_state = self._sim.get_agent_state()
        ref_position = agent_state.position
        ref_rotation = agent_state.rotation

        direction_vector = (
            np.array(episode.goals[0].position, dtype=np.float32)
            - ref_position
        )
        rotation_world_agent = quaternion_to_rotation(
            ref_rotation[3], ref_rotation[0], ref_rotation[1], ref_rotation[2]
        )
        direction_vector_agent = np.dot(
            rotation_world_agent.T, direction_vector
        )

        if self._goal_format == "POLAR":
            rho, phi = cartesian_to_polar(
                -direction_vector_agent[2], direction_vector_agent[0]
            )
            direction_vector_agent = np.array([rho, -phi], dtype=np.float32)

        return direction_vector_agent


class HeadingSensor(habitat.Sensor):
    """
       Sensor for observing the agent's heading in the global coordinate frame.

       Args:
           sim: reference to the simulator for calculating task observations.
           config: config for the sensor.
       """

    def __init__(self, sim: Simulator, config: Config):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "heading"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.HEADING

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float)

    def get_observation(self, observations, episode):
        agent_state = self._sim.get_agent_state()
        # Quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation

        direction_vector = np.array([0, 0, -1])

        rotation_world_agent = quaternion_to_rotation(
            ref_rotation[3], ref_rotation[0], ref_rotation[1], ref_rotation[2]
        )

        heading_vector = np.dot(rotation_world_agent.T, direction_vector)

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array(phi)


class ProximitySensor(habitat.Sensor):
    """
       Sensor for observing the distance to the closest obstacle

       Args:
           sim: reference to the simulator for calculating task observations.
           config: config for the sensor.
       """

    def __init__(self, sim, config):
        self._sim = sim
        self._max_detection_radius = getattr(
            config, "MAX_DETECTION_RADIUS", 2.0
        )
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "proximity"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0.0,
            high=self._max_detection_radius,
            shape=(1,),
            dtype=np.float,
        )

    def get_observation(self, observations, episode):
        current_position = self._sim.get_agent_state().position

        return self._sim.distance_to_closest_obstacle(
            current_position, self._max_detection_radius
        )


class SPL(habitat.Measure):
    """SPL (Success weighted by Path Length)

    ref: On Evaluation of Embodied Agents - Anderson et. al
    https://arxiv.org/pdf/1807.06757.pdf
    """

    def __init__(self, sim: Simulator, config: Config):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "spl"

    def reset_metric(self, episode):
        self._previous_position = self._sim.get_agent_state().position.tolist()
        self._start_end_episode_distance = episode.info["geodesic_distance"]
        self._agent_episode_distance = 0.0
        self._metric = None

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(self, episode, action):
        ep_success = 0
        current_position = self._sim.get_agent_state().position.tolist()

        distance_to_target = self._sim.geodesic_distance(
            current_position, episode.goals[0].position
        )

        if (
            action == self._sim.index_stop_action
            and distance_to_target < self._config.SUCCESS_DISTANCE
        ):
            ep_success = 1

        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = ep_success * (
            self._start_end_episode_distance
            / max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )


class Collisions(habitat.Measure):
    def __init__(self, sim, config):
        self._sim = sim
        self._config = config
        self._metric = None

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "collisions"

    def reset_metric(self, episode):
        self._metric = None

    def update_metric(self, episode, action):
        if self._metric is None:
            self._metric = 0

        current_position = self._sim.get_agent_state().position
        if (
            action == self._sim.index_forward_action
            and self._sim.distance_to_closest_obstacle(current_position)
            < COLLISION_PROXIMITY_TOLERANCE
        ):
            self._metric += 1


class NavigationTask(habitat.EmbodiedTask):
    def __init__(
        self,
        task_config: Config,
        sim: Simulator,
        dataset: Optional[Dataset] = None,
    ) -> None:

        task_measurements = []
        for measurement_name in task_config.MEASUREMENTS:
            measurement_cfg = getattr(task_config, measurement_name)
            is_valid_measurement = hasattr(
                habitat.tasks.nav.nav_task,  # type: ignore
                measurement_cfg.TYPE,
            )
            assert is_valid_measurement, "invalid measurement type {}".format(
                measurement_cfg.TYPE
            )
            task_measurements.append(
                getattr(
                    habitat.tasks.nav.nav_task,  # type: ignore
                    measurement_cfg.TYPE,
                )(sim, measurement_cfg)
            )
        self.measurements = Measurements(task_measurements)

        task_sensors = []
        for sensor_name in task_config.SENSORS:
            sensor_cfg = getattr(task_config, sensor_name)
            is_valid_sensor = hasattr(
                habitat.tasks.nav.nav_task, sensor_cfg.TYPE  # type: ignore
            )
            assert is_valid_sensor, "invalid sensor type {}".format(
                sensor_cfg.TYPE
            )
            task_sensors.append(
                getattr(
                    habitat.tasks.nav.nav_task, sensor_cfg.TYPE  # type: ignore
                )(sim, sensor_cfg)
            )

        self.sensor_suite = SensorSuite(task_sensors)
        super().__init__(config=task_config, sim=sim, dataset=dataset)

    def overwrite_sim_config(
        self, sim_config: Any, episode: Type[Episode]
    ) -> Any:
        return merge_sim_episode_config(sim_config, episode)
