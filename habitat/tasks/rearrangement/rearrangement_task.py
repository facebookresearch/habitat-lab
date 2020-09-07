#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
from typing import Any, Dict, List, Type

import attr
import numpy as np
from gym import spaces

from habitat.config.default import Config
from habitat.core.dataset import Episode
from habitat.core.embodied_task import Measure, SimulatorTaskAction
from habitat.core.registry import registry
from habitat.core.simulator import Observations, Sensor, SensorTypes, Simulator
from habitat.core.utils import not_none_validator
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.sims.rearrangement.actions import raycast
from habitat.sims.rearrangement.rearrangement_simulator import RearrangementSim
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationTask,
    PointGoalSensor,
    merge_sim_episode_config,
)
from habitat_sim.physics import MotionType
from habitat_sim.utils.common import quat_from_coeffs, quat_to_magnum


def merge_sim_episode_with_object_config(
    sim_config: Config, episode: Type[Episode]
) -> Any:
    sim_config = merge_sim_episode_config(sim_config, episode)
    sim_config.defrost()

    object_templates = {}
    for template in episode.object_templates:
        object_templates[template["object_key"]] = template["object_template"]

    objects = []
    for obj in episode.objects:
        obj.object_template = object_templates[obj.object_key]
        objects.append(obj)
    sim_config.objects = objects

    sim_config.freeze()

    return sim_config


@attr.s(auto_attribs=True, kw_only=True)
class RearrangementSpec:
    r"""Specifications that capture the initial or final pose of the object."""
    bbox: List[List[float]] = attr.ib(default=None)
    position: List[float] = attr.ib(default=None, validator=not_none_validator)
    rotation: List[float] = attr.ib(default=None, validator=not_none_validator)
    scale: List[float] = attr.ib(
        default=[1.0, 1.0, 1.0], validator=not_none_validator
    )


@attr.s(auto_attribs=True, kw_only=True)
class RearrangementObjectSpec(RearrangementSpec):
    r"""Object specifications that capture position of each object in the scene,
    the associated object template.
    """

    object_id: str = attr.ib(default=None, validator=not_none_validator)
    object_handle: str = attr.ib(default=None, validator=not_none_validator)


@attr.s(auto_attribs=True, kw_only=True)
class RearrangementEpisode(NavigationEpisode):
    r"""Specification of rearrangement episode.

    Args:
        episode_id: id of episode in the dataset
        scene_id: id of scene inside the simulator.
        start_position: numpy ndarray containing 3 entries for (x, y, z).
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation.
        goals: list of goals specifications
        objects: list of object initial spawn location and orientation.
        object_templates: list of all object templates used in this dataset.
    """

    objects: List[RearrangementObjectSpec] = attr.ib(
        default=None, validator=not_none_validator
    )
    goals: List[RearrangementSpec] = attr.ib(
        default=None, validator=not_none_validator
    )


@registry.register_measure
class ObjectToGoalDistance(Measure):
    """The measure calculates distance of object towards the goal."""

    cls_uuid: str = "object_to_goal_distance"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return ObjectToGoalDistance.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self.update_metric(*args, episode=episode, **kwargs)

    def _geo_dist(self, src_pos, goal_pos: np.array) -> float:
        return self._sim.geodesic_distance(src_pos, [goal_pos])

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(self, episode, *args: Any, **kwargs: Any):
        distance_to_target = {}

        for sim_obj_id in self._sim.get_existing_object_ids():
            obj_id = self._sim.sim_object_to_objid_mapping[sim_obj_id]

            previous_position = np.array(
                self._sim.get_translation(sim_obj_id)
            ).tolist()

            goal_position = episode.goals[obj_id].position

            distance_to_target[obj_id] = self._euclidean_distance(
                previous_position, goal_position
            )

        self._metric = distance_to_target


@registry.register_measure
class AgentToObjectDistance(Measure):
    """The measure calculates the distance of objects from the agent"""

    cls_uuid: str = "agent_to_object_distance"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return AgentToObjectDistance.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self.update_metric(*args, episode=episode, **kwargs)

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(self, episode, *args: Any, **kwargs: Any):
        distance_to_target = {}
        for sim_obj_id in enumerate(self._sim.get_existing_object_ids()):
            obj_id = self._sim.sim_object_to_objid_mapping[sim_obj_id]
            previous_position = np.array(
                self._sim.get_translation(sim_obj_id)
            ).tolist()
            agent_state = self._sim.get_agent_state()
            agent_position = agent_state.position

            distance_to_target[obj_id] = self._euclidean_distance(
                previous_position, agent_position
            )

        self._metric = distance_to_target


@registry.register_sensor
class GrippedObjectSensor(Sensor):
    def __init__(
        self, *args: Any, sim: RearrangementSim, config: Config, **kwargs: Any
    ):
        self._sim = sim
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Discrete(self._sim.get_existing_object_ids())

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "gripped_object_id"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.MEASUREMENT

    def get_observation(
        self,
        observations: Dict[str, Observations],
        episode: Episode,
        *args: Any,
        **kwargs: Any
    ):
        obj_id = self._sim.sim_object_to_objid_mapping.get(
            self._sim.gripped_object_id, -1
        )
        return obj_id


@registry.register_sensor
class AllObjectPositions(PointGoalSensor):
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "all_object_positions"

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (
            5,
            self._dimensionality,
        )

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(
        self, *args: Any, observations, episode, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        rotation_world_agent = agent_state.rotation
        sensor_data = np.zeros((5, 2))

        for sim_obj_id in enumerate(self._sim.get_existing_object_ids()):
            obj_id = self._sim.sim_object_to_objid_mapping[sim_obj_id]
            object_position = self._sim.get_translation(sim_obj_id)
            sensor_data[obj_id] = self._compute_pointgoal(
                agent_position, rotation_world_agent, object_position
            )

        return sensor_data


@registry.register_sensor
class AllObjectGoals(PointGoalSensor):
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "all_object_goals"

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (
            5,
            self._dimensionality,
        )

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(
        self, *args: Any, observations, episode, **kwargs: Any
    ):
        sensor_data = np.zeros((5, 2))
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        rotation_world_agent = agent_state.rotation

        for i, _ in enumerate(self._sim.get_existing_object_ids()):
            goal_position = np.array(
                episode.goals[i].position, dtype=np.float32
            )
            sensor_data[i] = self._compute_pointgoal(
                agent_position, rotation_world_agent, goal_position
            )

        return sensor_data


# Define actions here
@registry.register_task_action
class GrabOrReleaseAction(SimulatorTaskAction):
    def step(self, *args: Any, **kwargs: Any):
        r"""This method is called from ``Env`` on each ``step``."""

        gripped_object_id = self._sim._prev_sim_obs["gripped_object_id"]
        agent_config = self._default_agent.agent_config
        action_spec = agent_config.action_space[HabitatSimActions.GRAB_RELEASE]

        # If already holding an agent
        if gripped_object_id != -1:
            agent_body_transformation = (
                self._sim._default_agent.scene_node.transformation
            )
            T = np.dot(agent_body_transformation, self._sim.grip_offset)

            self._sim.set_transformation(T, gripped_object_id)

            position = self._sim.get_translation(gripped_object_id)

            if self._sim.pathfinder.is_navigable(position):
                self._sim.set_object_motion_type(
                    MotionType.STATIC, gripped_object_id
                )
                gripped_object_id = -1
                self._sim.recompute_navmesh(
                    self._sim.pathfinder, self._sim.navmesh_settings, True
                )
        # if not holding an object, then try to grab
        else:
            gripped_object_id = raycast(
                self._sim,
                action_spec.actuation.visual_sensor_name,
                crosshair_pos=action_spec.actuation.crosshair_pos,
                max_distance=action_spec.actuation.amount,
            )

            # found a grabbable object.
            if gripped_object_id != -1:
                agent_body_transformation = (
                    self._sim._default_agent.scene_node.transformation
                )

                self._sim.grip_offset = np.dot(
                    np.array(agent_body_transformation.inverted()),
                    np.array(self._sim.get_transformation(gripped_object_id)),
                )
                self._sim.set_object_motion_type(
                    MotionType.KINEMATIC, gripped_object_id
                )
                self._sim.recompute_navmesh(
                    self._sim.pathfinder, self._sim.navmesh_settings, True
                )

        # step physics by dt
        super().step_world(1 / 60.0)

        # Sync the gripped object after the agent moves.
        self._sim._sync_gripped_object(gripped_object_id)

        # obtain observations
        self._sim._prev_sim_obs.update(self._sim.get_sensor_observations())
        self._sim._prev_sim_obs["gripped_object_id"] = gripped_object_id

        observations = self._sim._sensor_suite.get_observations(
            self._sim._prev_sim_obs
        )
        return observations


@registry.register_task(name="RearrangementTask-v0")
class RearrangementTask(NavigationTask):
    r"""Embodied Rearrangement Task
    Goal: An agent must place objects at their corresponding goal position.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def register_object_templates(self):
        r"""
        Register object temmplates from the dataset into the simulator
        """
        obj_attr_mgr = self._sim.get_object_template_manager()
        object_templates = self._dataset.object_templates
        for name, template_info in object_templates.items():
            name = os.path.basename(name).split(".")[0]
            obj_handle = obj_attr_mgr.get_file_template_handles(name)[0]
            obj_template = obj_attr_mgr.get_template_by_handle(obj_handle)
            obj_template.scale = np.array(template_info["scale"])
            obj_attr_mgr.register_template(obj_template)

    def _initialize_objects(self, episode: RearrangementEpisode):
        r"""
        Initialize the stage with the objects in the episode.
        """
        obj_attr_mgr = self._sim.get_object_template_manager()

        # first remove all existing objects
        existing_object_ids = self._sim.get_existing_object_ids()
        if len(existing_object_ids) > 0:
            for obj_id in existing_object_ids:
                self._sim.remove_object(obj_id)

        self.sim_object_to_objid_mapping = {}
        self.objid_to_sim_object_mapping = {}

        for obj in episode.objects:
            object_rot = obj.rotation
            object_handle = obj_attr_mgr.get_file_template_handles(
                obj.object_handle
            )[0]
            object_id = self._sim.add_object_by_handle(object_handle)
            self._sim.sim_object_to_objid_mapping[object_id] = obj.object_id
            self._sim.objid_to_sim_object_mapping[obj.object_id] = object_id

            self._sim.set_translation(obj.position, object_id)
            if isinstance(object_rot, list):
                object_rot = quat_from_coeffs(object_rot)
            object_rot = quat_to_magnum(object_rot)
            self._sim.set_rotation(object_rot, object_id)
            self._sim.set_object_motion_type(MotionType.STATIC, object_id)

        # Recompute the navmesh after placing all the objects.
        self._sim.recompute_navmesh(
            self._sim.pathfinder, self._sim.navmesh_settings, True
        )

    def reset(self, episode: Episode):
        self._initialize_objects(episode)
        return super().reset(episode)

    def overwrite_sim_config(self, sim_config, episode):
        sim_config = super().overwrite_sim_config(sim_config, episode)
        self.register_object_templates()
        return sim_config
