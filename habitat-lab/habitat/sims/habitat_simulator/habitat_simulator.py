#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Union,
    cast,
)

import magnum as mn
import numpy as np
from gym import spaces
from gym.spaces.box import Box
from omegaconf import DictConfig

import habitat_sim
from habitat.config.default import get_agent_config
from habitat.core.batch_rendering.env_batch_renderer_constants import (
    KEYFRAME_OBSERVATION_KEY,
    KEYFRAME_SENSOR_PREFIX,
)
from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    DepthSensor,
    Observations,
    RGBSensor,
    SemanticSensor,
    Sensor,
    SensorSuite,
    ShortestPathPoint,
    Simulator,
    VisualObservation,
)
from habitat.core.spaces import Space

if TYPE_CHECKING:
    from torch import Tensor

    from habitat.config.default_structured_configs import SimulatorConfig


def overwrite_config(
    config_from: Any,
    config_to: Any,
    ignore_keys: Optional[Set[str]] = None,
    trans_dict: Optional[Dict[str, Callable]] = None,
) -> None:
    r"""Takes Habitat Lab config and Habitat-Sim config structures. Overwrites
    Habitat-Sim config with Habitat Lab values, where a field name is present
    in lowercase. Mostly used to avoid :ref:`sim_cfg.field = hapi_cfg.FIELD`
    code.
    Args:
        config_from: Habitat Lab config node.
        config_to: Habitat-Sim config structure.
        ignore_keys: Optional set of keys to ignore in config_to
        trans_dict: A Dict of str, callable which can be used on any value that has a matching key if not in ignore_keys.
    """

    def if_config_to_lower(config):
        if isinstance(config, DictConfig):
            return {
                key.lower(): val
                for key, val in config.items()
                if isinstance(key, str)
            }
        else:
            return config

    for attr, value in config_from.items():
        assert isinstance(attr, str)
        low_attr = attr.lower()
        if ignore_keys is None or low_attr not in ignore_keys:
            if hasattr(config_to, low_attr):
                if trans_dict is not None and low_attr in trans_dict:
                    setattr(config_to, low_attr, trans_dict[low_attr](value))
                else:
                    setattr(config_to, low_attr, if_config_to_lower(value))
            else:
                raise NameError(
                    f"""{low_attr} is not found on habitat_sim but is found on habitat_lab config.
                    It's also not in the list of keys to ignore: {ignore_keys}
                    Did you make a typo in the config?
                    If not the version of Habitat Sim may not be compatible with Habitat Lab version: {config_from}
                    """
                )


class HabitatSimSensor:
    sim_sensor_type: habitat_sim.SensorType
    _get_default_spec = Callable[..., habitat_sim.sensor.SensorSpec]
    _config_ignore_keys = {"height", "type", "width"}


@registry.register_sensor
class HabitatSimRGBSensor(RGBSensor, HabitatSimSensor):
    _get_default_spec = habitat_sim.CameraSensorSpec
    sim_sensor_type = habitat_sim.SensorType.COLOR

    RGBSENSOR_DIMENSION = 3

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(
                self.config.height,
                self.config.width,
                self.RGBSENSOR_DIMENSION,
            ),
            dtype=np.uint8,
        )

    def get_observation(
        self, sim_obs: Dict[str, Union[np.ndarray, bool, "Tensor"]]
    ) -> VisualObservation:
        obs = cast(Optional[VisualObservation], sim_obs.get(self.uuid, None))
        check_sim_obs(obs, self)

        # remove alpha channel
        obs = obs[:, :, : self.RGBSENSOR_DIMENSION]  # type: ignore[index]
        return obs


@registry.register_sensor
class HabitatSimDepthSensor(DepthSensor, HabitatSimSensor):
    _get_default_spec = habitat_sim.CameraSensorSpec
    _config_ignore_keys = {
        "max_depth",
        "min_depth",
        "normalize_depth",
    }.union(HabitatSimSensor._config_ignore_keys)
    sim_sensor_type = habitat_sim.SensorType.DEPTH

    min_depth_value: float
    max_depth_value: float

    def __init__(self, config: DictConfig) -> None:
        self.min_depth_value = config.min_depth
        self.max_depth_value = config.max_depth
        self.normalize_depth = config.normalize_depth
        if self.normalize_depth:
            self._obs_shape = spaces.Box(
                low=0,
                high=1,
                shape=(config.height, config.width, 1),
                dtype=np.float32,
            )
        else:
            self._obs_shape = spaces.Box(
                low=self.min_depth_value,
                high=self.max_depth_value,
                shape=(config.height, config.width, 1),
                dtype=np.float32,
            )

        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Box:
        return self._obs_shape

    def get_observation(
        self, sim_obs: Dict[str, Union[np.ndarray, bool, "Tensor"]]
    ) -> VisualObservation:
        obs = cast(Optional[VisualObservation], sim_obs.get(self.uuid, None))
        check_sim_obs(obs, self)
        if isinstance(obs, np.ndarray):
            obs = np.clip(obs, self.min_depth_value, self.max_depth_value)

            obs = np.expand_dims(
                obs, axis=2
            )  # make depth observation a 3D array
        else:
            obs = obs.clamp(self.min_depth_value, self.max_depth_value)  # type: ignore[attr-defined, unreachable]

            obs = obs.unsqueeze(-1)  # type: ignore[attr-defined]

        if self.normalize_depth:
            # normalize depth observation to [0, 1]
            obs = (obs - self.min_depth_value) / (
                self.max_depth_value - self.min_depth_value
            )

        return obs


@registry.register_sensor
class HabitatSimSemanticSensor(SemanticSensor, HabitatSimSensor):
    _get_default_spec = habitat_sim.CameraSensorSpec
    sim_sensor_type = habitat_sim.SensorType.SEMANTIC

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.iinfo(np.uint32).min,
            high=np.iinfo(np.uint32).max,
            shape=(self.config.height, self.config.width, 1),
            dtype=np.int32,
        )

    def get_observation(
        self, sim_obs: Dict[str, Union[np.ndarray, bool, "Tensor"]]
    ) -> VisualObservation:
        obs = cast(Optional[VisualObservation], sim_obs.get(self.uuid, None))
        check_sim_obs(obs, self)
        # make semantic observation a 3D array
        if isinstance(obs, np.ndarray):
            obs = obs[..., None].astype(np.int32)
        else:
            obs = obs[..., None]
        return obs


# TODO Sensor Hierarchy needs to be redone here. These should not subclass camera sensors
@registry.register_sensor
class HabitatSimEquirectangularRGBSensor(HabitatSimRGBSensor):
    _get_default_spec = habitat_sim.EquirectangularSensorSpec


@registry.register_sensor
class HabitatSimEquirectangularDepthSensor(HabitatSimDepthSensor):
    _get_default_spec = habitat_sim.EquirectangularSensorSpec


@registry.register_sensor
class HabitatSimEquirectangularSemanticSensor(HabitatSimSemanticSensor):
    _get_default_spec = habitat_sim.EquirectangularSensorSpec


@registry.register_sensor
class HabitatSimFisheyeRGBSensor(HabitatSimRGBSensor):
    _get_default_spec = habitat_sim.FisheyeSensorDoubleSphereSpec


@registry.register_sensor
class HabitatSimFisheyeDepthSensor(HabitatSimDepthSensor):
    _get_default_spec = habitat_sim.FisheyeSensorDoubleSphereSpec


@registry.register_sensor
class HabitatSimFisheyeSemanticSensor(HabitatSimSemanticSensor):
    _get_default_spec = habitat_sim.FisheyeSensorDoubleSphereSpec


def check_sim_obs(
    obs: Union[np.ndarray, "Tensor", None], sensor: Sensor
) -> None:
    assert obs is not None, (
        "Observation corresponding to {} not present in "
        "simulator's observations".format(sensor.uuid)
    )


@registry.register_simulator(name="Sim-v0")
class HabitatSim(habitat_sim.Simulator, Simulator):
    r"""Simulator wrapper over habitat-sim

    habitat-sim repo: https://github.com/facebookresearch/habitat-sim

    Args:
        config: configuration for initializing the simulator.
    """

    def __init__(self, config: "SimulatorConfig") -> None:
        self.habitat_config = config

        sim_sensors = []
        for agent_config in self.habitat_config.agents.values():
            for sensor_cfg in agent_config.sim_sensors.values():
                sensor_type = registry.get_sensor(sensor_cfg.type)

                assert (
                    sensor_type is not None
                ), "invalid sensor type {}".format(sensor_cfg.type)
                sim_sensors.append(sensor_type(sensor_cfg))

        self._sensor_suite = SensorSuite(sim_sensors)
        self.sim_config = self.create_sim_config(self._sensor_suite)
        self._current_scene = self.sim_config.sim_cfg.scene_id
        super().__init__(self.sim_config)
        # load additional object paths specified by the dataset
        # TODO: Should this be moved elsewhere?
        obj_attr_mgr = self.get_object_template_manager()
        for path in self.habitat_config.additional_object_paths:
            obj_attr_mgr.load_configs(path)
        self._action_space = spaces.Discrete(
            len(
                self.sim_config.agents[
                    self.habitat_config.default_agent_id
                ].action_space
            )
        )
        self._prev_sim_obs: Optional[Observations] = None

    def create_sim_config(
        self, _sensor_suite: SensorSuite
    ) -> habitat_sim.Configuration:
        sim_config = habitat_sim.SimulatorConfiguration()
        # Check if Habitat-Sim is post Scene Config Update
        if not hasattr(sim_config, "scene_id"):
            raise RuntimeError(
                "Incompatible version of Habitat-Sim detected, please upgrade habitat_sim"
            )
        overwrite_config(
            config_from=self.habitat_config.habitat_sim_v0,
            config_to=sim_config,
            # Ignore key as it gets propagated to sensor below
            ignore_keys={"gpu_gpu"},
        )
        sim_config.scene_dataset_config_file = (
            self.habitat_config.scene_dataset
        )
        sim_config.scene_id = self.habitat_config.scene
        lab_agent_config = get_agent_config(self.habitat_config)
        agent_config = habitat_sim.AgentConfiguration()
        overwrite_config(
            config_from=lab_agent_config,
            config_to=agent_config,
            # These keys are only used by Hab-Lab
            ignore_keys={
                "is_set_start_state",
                # This is the Sensor Config. Unpacked below
                "sensors",
                "sim_sensors",
                "start_position",
                "start_rotation",
                "articulated_agent_urdf",
                "articulated_agent_type",
                "joint_start_noise",
                "joint_that_can_control",
                "motion_data_path",
                "ik_arm_urdf",
                "grasp_managers",
                "max_climb",
                "max_slope",
                "joint_start_override",
                "auto_update_sensor_transform",
            },
        )

        # configure default navmesh parameters to match the configured agent
        if self.habitat_config.default_agent_navmesh:
            sim_config.navmesh_settings = habitat_sim.nav.NavMeshSettings()
            sim_config.navmesh_settings.set_defaults()
            sim_config.navmesh_settings.agent_radius = agent_config.radius
            sim_config.navmesh_settings.agent_height = agent_config.height
            sim_config.navmesh_settings.agent_max_climb = (
                lab_agent_config.max_climb
            )
            sim_config.navmesh_settings.agent_max_slope = (
                lab_agent_config.max_slope
            )
            sim_config.navmesh_settings.include_static_objects = (
                self.habitat_config.navmesh_include_static_objects
            )

        sensor_specifications = []
        for sensor in _sensor_suite.sensors.values():
            assert isinstance(sensor, HabitatSimSensor)
            sim_sensor_cfg = sensor._get_default_spec()  # type: ignore[operator]
            overwrite_config(
                config_from=sensor.config,
                config_to=sim_sensor_cfg,
                # These keys are only used by Hab-Lab
                # or translated into the sensor config manually
                ignore_keys=sensor._config_ignore_keys,
                # TODO consider making trans_dict a sensor class var too.
                trans_dict={
                    "sensor_model_type": lambda v: getattr(
                        habitat_sim.FisheyeSensorModelType, v
                    ),
                    "sensor_subtype": lambda v: getattr(
                        habitat_sim.SensorSubType, v
                    ),
                },
            )
            sim_sensor_cfg.uuid = sensor.uuid
            sim_sensor_cfg.resolution = list(
                sensor.observation_space.shape[:2]
            )

            # TODO(maksymets): Add configure method to Sensor API to avoid
            # accessing child attributes through parent interface
            # We know that the Sensor has to be one of these Sensors
            sim_sensor_cfg.sensor_type = sensor.sim_sensor_type
            sim_sensor_cfg.gpu2gpu_transfer = (
                self.habitat_config.habitat_sim_v0.gpu_gpu
            )
            sensor_specifications.append(sim_sensor_cfg)

        agent_config.sensor_specifications = sensor_specifications

        agent_config.action_space = {
            0: habitat_sim.ActionSpec("stop"),
            1: habitat_sim.ActionSpec(
                "move_forward",
                habitat_sim.ActuationSpec(
                    amount=self.habitat_config.forward_step_size
                ),
            ),
            2: habitat_sim.ActionSpec(
                "turn_left",
                habitat_sim.ActuationSpec(
                    amount=self.habitat_config.turn_angle
                ),
            ),
            3: habitat_sim.ActionSpec(
                "turn_right",
                habitat_sim.ActuationSpec(
                    amount=self.habitat_config.turn_angle
                ),
            ),
        }

        output = habitat_sim.Configuration(sim_config, [agent_config])
        output.enable_batch_renderer = (
            self.habitat_config.renderer.enable_batch_renderer
        )
        return output

    @property
    def sensor_suite(self) -> SensorSuite:
        return self._sensor_suite

    @property
    def action_space(self) -> Space:
        return self._action_space

    def _update_agents_state(self) -> bool:
        is_updated = False
        for agent_id, agent_name in enumerate(
            self.habitat_config.agents_order
        ):
            agent_cfg = self.habitat_config.agents[agent_name]
            if agent_cfg.is_set_start_state:
                self.set_agent_state(
                    [float(k) for k in agent_cfg.start_position],
                    [float(k) for k in agent_cfg.start_rotation],
                    agent_id,
                )
                is_updated = True

        return is_updated

    def reset(self) -> Observations:
        sim_obs = super().reset()
        if self._update_agents_state():
            sim_obs = self.get_sensor_observations()

        self._prev_sim_obs = sim_obs
        if self.config.enable_batch_renderer:
            self.add_keyframe_to_observations(sim_obs)
            return sim_obs
        else:
            return self._sensor_suite.get_observations(sim_obs)

    def step(
        self, action: Optional[Union[str, np.ndarray, int]]
    ) -> Observations:
        if action is None:
            sim_obs = self.get_sensor_observations()
        else:
            sim_obs = super().step(action)
        self._prev_sim_obs = sim_obs
        if self.config.enable_batch_renderer:
            self.add_keyframe_to_observations(sim_obs)
            return sim_obs
        else:
            return self._sensor_suite.get_observations(sim_obs)

    def render(self, mode: str = "rgb") -> Any:
        r"""
        Args:
            mode: sensor whose observation is used for returning the frame,
                eg: "rgb", "depth", "semantic"

        Returns:
            rendered frame according to the mode
        """
        assert not self.config.enable_batch_renderer

        sim_obs = self.get_sensor_observations()
        observations = self._sensor_suite.get_observations(sim_obs)

        output = observations.get(mode)
        assert output is not None, "mode {} sensor is not active".format(mode)
        if not isinstance(output, np.ndarray):
            # If it is not a numpy array, it is a torch tensor
            # The function expects the result to be a numpy array
            output = output.to("cpu").numpy()

        return output

    def reconfigure(
        self,
        habitat_config: "SimulatorConfig",
        ep_info: Optional[Episode] = None,
        should_close_on_new_scene: bool = True,
    ) -> None:
        # TODO(maksymets): Switch to Habitat-Sim more efficient caching
        is_same_scene = habitat_config.scene == self._current_scene
        self.habitat_config = habitat_config
        self.sim_config = self.create_sim_config(self._sensor_suite)
        if not is_same_scene:
            self._current_scene = habitat_config.scene
            if should_close_on_new_scene:
                self.close(destroy=False)
            super().reconfigure(self.sim_config)

        self._update_agents_state()

    def geodesic_distance(
        self,
        position_a: Union[Sequence[float], np.ndarray],
        position_b: Union[
            Sequence[float], Sequence[Sequence[float]], np.ndarray
        ],
        episode: Optional[Episode] = None,
    ) -> float:
        if episode is None or episode._shortest_path_cache is None:
            path = habitat_sim.MultiGoalShortestPath()
            if isinstance(position_b[0], (Sequence, np.ndarray)):
                path.requested_ends = np.array(position_b, dtype=np.float32)
            else:
                path.requested_ends = np.array(
                    [np.array(position_b, dtype=np.float32)]
                )
        else:
            path = episode._shortest_path_cache

        path.requested_start = np.array(position_a, dtype=np.float32)

        self.pathfinder.find_path(path)

        if episode is not None:
            episode._shortest_path_cache = path

        return path.geodesic_distance

    def action_space_shortest_path(
        self,
        source: AgentState,
        targets: Sequence[AgentState],
        agent_id: int = 0,
    ) -> List[ShortestPathPoint]:
        r"""
        Returns:
            List of agent states and actions along the shortest path from
            source to the nearest target (both included). If one of the
            target(s) is identical to the source, a list containing only
            one node with the identical agent state is returned. Returns
            an empty list in case none of the targets are reachable from
            the source. For the last item in the returned list the action
            will be None.
        """
        raise NotImplementedError(
            "This function is no longer implemented. Please use the greedy "
            "follower instead"
        )

    @property
    def up_vector(self) -> np.ndarray:
        return np.array([0.0, 1.0, 0.0])

    @property
    def forward_vector(self) -> np.ndarray:
        return -np.array([0.0, 0.0, 1.0])

    def get_straight_shortest_path_points(self, position_a, position_b):
        path = habitat_sim.ShortestPath()
        path.requested_start = position_a
        path.requested_end = position_b
        self.pathfinder.find_path(path)
        return path.points

    def sample_navigable_point(self) -> List[float]:
        return list(self.pathfinder.get_random_navigable_point())

    def is_navigable(self, point: List[float]) -> bool:
        return self.pathfinder.is_navigable(point)

    def semantic_annotations(self):
        r"""
        Returns:
            SemanticScene which is a three level hierarchy of semantic
            annotations for the current scene. Specifically this method
            returns a SemanticScene which contains a list of SemanticLevel's
            where each SemanticLevel contains a list of SemanticRegion's where
            each SemanticRegion contains a list of SemanticObject's.

            SemanticScene has attributes: aabb(axis-aligned bounding box) which
            has attributes aabb.center and aabb.sizes which are 3d vectors,
            categories, levels, objects, regions.

            SemanticLevel has attributes: id, aabb, objects and regions.

            SemanticRegion has attributes: id, level, aabb, category (to get
            name of category use category.name()) and objects.

            SemanticObject has attributes: id, region, aabb, obb (oriented
            bounding box) and category.

            SemanticScene contains List[SemanticLevels]
            SemanticLevel contains List[SemanticRegion]
            SemanticRegion contains List[SemanticObject]

            Example to loop through in a hierarchical fashion:
            for level in semantic_scene.levels:
                for region in level.regions:
                    for obj in region.objects:
        """
        return self.semantic_scene

    def get_agent_state(self, agent_id: int = 0) -> habitat_sim.AgentState:
        return self.get_agent(agent_id).get_state()

    def set_agent_state(
        self,
        position: List[float],
        rotation: List[float],
        agent_id: int = 0,
        reset_sensors: bool = True,
    ) -> bool:
        r"""Sets agent state similar to initialize_agent, but without agents
        creation. On failure to place the agent in the proper position, it is
        moved back to its previous pose.

        Args:
            position: list containing 3 entries for (x, y, z).
            rotation: list with 4 entries for (x, y, z, w) elements of unit
                quaternion (versor) representing agent 3D orientation,
                (https://en.wikipedia.org/wiki/Versor)
            agent_id: int identification of agent from multiagent setup.
            reset_sensors: bool for if sensor changes (e.g. tilt) should be
                reset).

        Returns:
            True if the set was successful else moves the agent back to its
            original pose and returns false.
        """
        agent = self.get_agent(agent_id)
        new_state = self.get_agent(agent_id).get_state()
        new_state.position = position
        new_state.rotation = rotation

        # NB: The agent state also contains the sensor states in _absolute_
        # coordinates. In order to set the agent's body to a specific
        # location and have the sensors follow, we must not provide any
        # state for the sensors. This will cause them to follow the agent's
        # body
        new_state.sensor_states = {}
        agent.set_state(new_state, reset_sensors)
        return True

    def get_observations_at(
        self,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        keep_agent_at_new_pose: bool = False,
    ) -> Optional[Observations]:
        current_state = self.get_agent_state()
        if position is None or rotation is None:
            success = True
        else:
            success = self.set_agent_state(
                position, rotation, reset_sensors=False
            )

        if success:
            sim_obs = self.get_sensor_observations()

            self._prev_sim_obs = sim_obs

            observations = self._sensor_suite.get_observations(sim_obs)
            if not keep_agent_at_new_pose:
                self.set_agent_state(
                    current_state.position,
                    current_state.rotation,
                    reset_sensors=False,
                )
            return observations
        else:
            return None

    def distance_to_closest_obstacle(
        self, position: np.ndarray, max_search_radius: float = 2.0
    ) -> float:
        return self.pathfinder.distance_to_closest_obstacle(
            position, max_search_radius
        )

    def island_radius(self, position: Sequence[float]) -> float:
        return self.pathfinder.island_radius(position)

    @property
    def previous_step_collided(self):
        r"""Whether or not the previous step resulted in a collision

        Returns:
            bool: True if the previous step resulted in a collision, false otherwise

        Warning:
            This field is only updated when :meth:`step`, :meth:`reset`, or :meth:`get_observations_at` are
            called.  It does not update when the agent is moved to a new location.  Furthermore, it
            will _always_ be false after :meth:`reset` or :meth:`get_observations_at` as neither of those
            result in an action (step) being taken.
        """
        return self._prev_sim_obs.get("collided", False)

    def add_keyframe_to_observations(self, observations):
        r"""Adds an item to observations that contains the latest gfx-replay keyframe.
        This is used to communicate the state of concurrent simulators to the batch renderer between processes.

        :param observations: Original observations upon which the keyframe is added.
        """
        assert self.config.enable_batch_renderer

        assert KEYFRAME_OBSERVATION_KEY not in observations
        for _sensor_uuid, sensor in self._sensors.items():
            node = sensor._sensor_object.node
            transform = node.absolute_transformation()
            rotation = mn.Quaternion.from_matrix(transform.rotation())
            self.gfx_replay_manager.add_user_transform_to_keyframe(
                KEYFRAME_SENSOR_PREFIX + _sensor_uuid,
                transform.translation,
                rotation,
            )
        observations[
            KEYFRAME_OBSERVATION_KEY
        ] = self.gfx_replay_manager.extract_keyframe()
