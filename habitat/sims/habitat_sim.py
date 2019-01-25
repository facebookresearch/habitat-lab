from enum import Enum
from typing import List, Any, Dict, Optional

import esp as habitat_sim
import numpy as np
from gym import spaces

import habitat
from habitat import SensorSuite
from habitat.core.logging import logger
from habitat.core.simulator import RGBSensor, DepthSensor, SemanticSensor
from habitat.core.simulator import AgentState, ShortestPathPoint


UUID_RGBSENSOR = "rgb"
# Sim provides RGB as RGBD structure with 4 dimensions
RGBSENSOR_DIMENSION = 4

DEPTHSENSOR_MIN_DEPTH = 0
DEPTHSENSOR_MAX_DEPTH = 10
DEPTHSENSOR_NORMALIZE = True


def overwrite_config(config_from: Dict, config_to) -> None:
    for attr, value in config_from.items():
        setattr(config_to, attr, value)


def check_sim_obs(obs, sensor):
    assert obs is not None, (
        "Observation corresponding to {} not present in "
        "simulator's observations".format(sensor.uuid)
    )


class HabitatSimRGBSensor(RGBSensor):
    """RGB sensor for habitat_sim
    """

    def __init__(self, config):
        self.sim_sensor_type = habitat_sim.SensorType.COLOR
        super().__init__(config)

    def _get_observation_space(self, config):
        return spaces.Box(
            low=0,
            high=255,
            shape=(config.height, config.width, RGBSENSOR_DIMENSION),
            dtype=np.uint8,
        )

    def get_observation(self, sim_obs):
        obs = sim_obs.get(self.uuid, None)
        check_sim_obs(obs, self)
        return obs


class HabitatSimDepthSensor(DepthSensor):
    """Depth sensor for habitat_sim
    """

    def __init__(self, config):
        self.sim_sensor_type = habitat_sim.SensorType.DEPTH
        self.min_depth = DEPTHSENSOR_MIN_DEPTH
        if hasattr(config, "min_depth"):
            self.min_depth = config.min_depth

        self.max_depth = DEPTHSENSOR_MAX_DEPTH
        if hasattr(config, "max_depth"):
            self.max_depth = config.max_depth

        self.normalize_depth = DEPTHSENSOR_NORMALIZE
        if hasattr(config, "normalize_depth"):
            self.normalize_depth = config.normalize_depth

        if self.normalize_depth:
            self.min_depth_value = 0
            self.max_depth_value = 1
        else:
            self.min_depth_value = self.min_depth
            self.max_depth_value = self.max_depth

        super().__init__(config)

    def _get_observation_space(self, config):
        return spaces.Box(
            low=self.min_depth_value,
            high=self.max_depth_value,
            shape=(config.height, config.width),
            dtype=np.float32,
        )

    def get_observation(self, sim_obs):
        obs = sim_obs.get(self.uuid, None)
        check_sim_obs(obs, self)

        obs = np.clip(obs, self.min_depth, self.max_depth)
        if self.normalize_depth:
            # normalize depth observation to [0, 1]
            obs = (obs - self.min_depth) / self.max_depth

        return obs


class HabitatSimSemanticSensor(SemanticSensor):
    """Semantic sensor for habitat_sim
    """

    def __init__(self, config):
        self.sim_sensor_type = habitat_sim.SensorType.SEMANTIC
        super().__init__(config)

    def _get_observation_space(self, config):
        return spaces.Box(
            low=np.iinfo(np.uint32).min,
            high=np.iinfo(np.uint32).max,
            shape=(config.height, config.width),
            dtype=np.uint32,
        )

    def get_observation(self, sim_obs):
        obs = sim_obs.get(self.uuid, None)
        check_sim_obs(obs, self)
        return obs


class SimActions(Enum):
    LEFT = "look_left"
    RIGHT = "look_right"
    FORWARD = "move_forward"
    STOP = "stop"


SIM_ACTION_TO_NAME = {
    0: SimActions.FORWARD.value,
    1: SimActions.LEFT.value,
    2: SimActions.RIGHT.value,
    3: SimActions.STOP.value,
}

SIM_NAME_TO_ACTION = {
    SimActions.FORWARD.value: 0,
    SimActions.LEFT.value: 1,
    SimActions.RIGHT.value: 2,
    SimActions.STOP.value: 3,
}


class HabitatSim(habitat.Simulator):
    def __init__(self, config: Any) -> None:
        self.config = config.clone()

        sim_sensors = []
        for s in config.sensors:
            is_valid_sensor = hasattr(
                habitat.sims.habitat_sim, s
            )  # type: ignore
            assert is_valid_sensor, "invalid sensor type {}".format(s)
            sim_sensors.append(
                getattr(habitat.sims.habitat_sim, s)(config)
            )  # type: ignore

        self.sensor_suite = SensorSuite(sim_sensors)

        self.sim_config = HabitatSim.create_sim_config(
            config, self.sensor_suite
        )

        self._sim = habitat_sim.Simulator(self.sim_config)

        self.action_space = spaces.Discrete(
            len(self.sim_config.agents[0].action_space)
        )

        self.episode_active = False
        self._controls = SIM_ACTION_TO_NAME

    @staticmethod
    def create_sim_config(
        config: Any, sensor_suite: SensorSuite
    ) -> habitat_sim.SimulatorConfiguration:
        sim_config = habitat_sim.SimulatorConfiguration()
        # TODO(maksymets): use general notion of scene from Resource Manager
        sim_config.scene.id = config.scene
        sim_config.gpu_device_id = config.gpu_device_id
        agent_config = habitat_sim.AgentConfiguration()
        overwrite_config(
            config_from=config.agents[config.default_agent_id],
            config_to=agent_config,
        )
        sensor_specifications = []
        for sensor in sensor_suite.sensors.values():
            sim_sensor_config = habitat_sim.SensorSpec()
            sim_sensor_config.uuid = sensor.uuid
            sim_sensor_config.resolution = list(
                sensor.observation_space.shape[:2]
            )
            sim_sensor_config.parameters["hfov"] = config.hfov
            sim_sensor_config.position = config.sensor_position
            sim_sensor_config.sensor_type = (
                sensor.sim_sensor_type
            )  # type: ignore
            sensor_specifications.append(sim_sensor_config)

        agent_config.sensor_specifications = sensor_specifications
        agent_config.action_space = {
            SimActions.LEFT.value: habitat_sim.ActionSpec(
                "lookLeft", {"amount": config.turn_angle}
            ),
            SimActions.RIGHT.value: habitat_sim.ActionSpec(
                "lookRight", {"amount": config.turn_angle}
            ),
            SimActions.FORWARD.value: habitat_sim.ActionSpec(
                "moveForward", {"amount": config.forward_step_size}
            ),
            SimActions.STOP.value: habitat_sim.ActionSpec("stop", {}),
        }
        sim_config.agents = [agent_config]
        return sim_config

    def reset(self):
        sim_obs = self._sim.reset()

        if (
            hasattr(self.config, "start_position")
            and hasattr(self.config, "start_rotation")
            and self.config.start_position is not None
            and self.config.start_rotation is not None
        ):
            self.set_agent_state(
                self.config.start_position,
                self.config.start_rotation,
                self.config.default_agent_id,
            )
            sim_obs = self._sim.get_sensor_observations()
        self.episode_active = True
        return self.sensor_suite.get_observations(sim_obs)

    def step(self, action):
        assert self.episode_active, (
            "episode is not active, environment not RESET or "
            "STOP action called previously"
        )
        sim_action = self._controls[action]
        if sim_action == SimActions.STOP.value:
            # TODO(akadian): Handle reward calculation on stop once pointnav
            # is integrated
            self.episode_active = False
            sim_obs = self._sim.get_sensor_observations()
        else:
            sim_obs = self._sim.step(sim_action)
        observations = self.sensor_suite.get_observations(sim_obs)
        return observations

    def render(self):
        return self._sim.render()

    def seed(self, seed):
        self._sim.seed(seed)

    def reconfigure(self, config: Any) -> None:
        # TODO(maksymets): Switch to Habitat-Sim more efficient caching
        is_same_scene = config.scene == self.config.scene
        self.config = config.clone()
        self.sim_config = self.create_sim_config(config, self.sensor_suite)
        if is_same_scene:
            self._sim.reconfigure(self.sim_config)
        else:
            self._sim.close()
            del self._sim
            self._sim = habitat_sim.Simulator(self.sim_config)

        if (
            hasattr(config, "start_position")
            and hasattr(config, "start_rotation")
            and config.start_position is not None
            and config.start_rotation is not None
        ):
            self.initialize_agent(
                config.start_position,
                config.start_rotation,
                self.config.default_agent_id,
            )

    def geodesic_distance(self, position_a, position_b):
        path = habitat_sim.ShortestPath()
        path.requested_start = position_a
        path.requested_end = position_b
        self._sim.pathfinder.find_path(path)
        return path.geodesic_distance

    def action_space_shortest_paths(
        self, source: AgentState, targets: List[AgentState], agent_id: int = 0
    ) -> List[ShortestPathPoint]:
        assert agent_id == 0, "No support of multi agent in {} yet.".format(
            self.__class__.__name__
        )
        action_pathfinder = self._sim.make_action_pathfinder(agent_id=agent_id)
        action_shortest_path = habitat_sim.MultiGoalActionSpaceShortestPath()
        action_shortest_path.requested_start.position = source.position
        action_shortest_path.requested_start.rotation = source.rotation

        for target in targets:
            action_shortest_path.requested_ends.append(
                habitat_sim.ActionSpacePathLocation(
                    target.position, target.rotation
                )
            )

        if not action_pathfinder.find_path(action_shortest_path):
            return []

        # add None action to last node in path
        actions: List[Optional[int]] = [
            SIM_NAME_TO_ACTION[action]
            for action in action_shortest_path.actions
        ]
        actions.append(None)

        shortest_path = [
            ShortestPathPoint(position, rotation, action)
            for position, rotation, action in zip(
                action_shortest_path.points,
                action_shortest_path.rotations,
                actions,
            )
        ]

        return shortest_path

    def sample_navigable_point(self):
        r"""
        return: randomly sample a [x, y, z] point where the agent can be
        initialized.
        """
        return self._sim.pathfinder.get_random_navigable_point()

    def semantic_annotations(self):
        r"""
        :return: SemanticScene which is a three level hierarchy of
        semantic annotations for the current scene. Specifically this method
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

        Example to loop through in a hierarchical fashion:
        for level in semantic_scene.levels:
            for region in level.regions:
                for obj in region.objects:
        """
        return self._sim.semantic_scene

    def close(self):
        self._sim.close()

    def agent_state(self, agent_id: int = 0):
        assert agent_id == 0, "No support of multi agent in {} yet.".format(
            self.__class__.__name__
        )
        state = habitat_sim.AgentState()
        self._sim.get_agent(agent_id).get_state(state)
        return state

    def set_agent_state(
        self,
        position: List[float] = None,
        rotation: List[float] = None,
        agent_id: int = 0,
    ) -> None:
        r"""
        Sets agent state similar to initialize_agent, but without agents
        creation.
        :param position: numpy ndarray containing 3 entries for (x, y, z)
        :param rotation: numpy ndarray with 4 entries for (x, y, z, w) elements
        of unit quaternion (versor) representing agent 3D orientation,
        ref: https://en.wikipedia.org/wiki/Versor
        :param agent_id: int identification of agent from multiagent setup
        """
        agent = self._sim.get_agent(agent_id)
        state = self.agent_state(agent_id)
        state.position = position
        state.rotation = rotation
        agent.set_state(state)

        self._check_agent_position(position, agent_id)

    def initialize_agent(self, position, rotation, agent_id=0):
        r"""
        :param position: numpy ndarray containing 3 entries for (x, y, z)
        :param rotation: numpy ndarray with 4 entries for (x, y, z, w) elements
        of unit quaternion (versor) representing agent 3D orientation,
        ref: https://en.wikipedia.org/wiki/Versor
        :param agent_id: int identification of agent from multiagent setup
        """
        agent_state = habitat_sim.AgentState()
        agent_state.position = position
        agent_state.rotation = rotation

        assert agent_id == 0, "No support of multi agent in {} yet.".format(
            self.__class__.__name__
        )
        self._sim.initialize_agent(
            agent_id=agent_id, initial_state=agent_state
        )
        self._check_agent_position(position, agent_id)

    # TODO (maksymets): Remove check after simulator became stable
    def _check_agent_position(self, position, agent_id=0):
        if not np.allclose(position, self.agent_state(agent_id).position):
            logger.info("Agent state diverges from configured start position.")
