from enum import Enum
from typing import List, Any, Dict, Optional

import habitat_sim
import numpy as np
from gym import spaces

import habitat
from habitat import SensorSuite
from habitat.core.logging import logger
from habitat.core.simulator import AgentState, ShortestPathPoint
from habitat.core.simulator import RGBSensor, DepthSensor, SemanticSensor

# Sim provides RGB as RGBD structure with 4 dimensions
RGBSENSOR_DIMENSION = 4


def overwrite_config(config_from: Dict, config_to) -> None:
    for attr, value in config_from.items():
        if hasattr(config_to, attr.lower()):
            setattr(config_to, attr.lower(), value)


def check_sim_obs(obs, sensor):
    assert obs is not None, (
        "Observation corresponding to {} not present in "
        "simulator's observations".format(sensor.uuid)
    )


class HabitatSimRGBSensor(RGBSensor):
    """RGB sensor for habitat_sim
    """

    sim_sensor_type: habitat_sim.SensorType

    def __init__(self, config):
        self.sim_sensor_type = habitat_sim.SensorType.COLOR
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.config.HEIGHT, self.config.WIDTH, RGBSENSOR_DIMENSION),
            dtype=np.uint8,
        )

    def get_observation(self, sim_obs):
        obs = sim_obs.get(self.uuid, None)
        check_sim_obs(obs, self)
        return obs


class HabitatSimDepthSensor(DepthSensor):
    """Depth sensor for habitat_sim
    """

    sim_sensor_type: habitat_sim.SensorType
    min_depth_value: float
    max_depth_value: float

    def __init__(self, config):
        self.sim_sensor_type = habitat_sim.SensorType.DEPTH

        if config.NORMALIZE_DEPTH:
            self.min_depth_value = 0
            self.max_depth_value = 1
        else:
            self.min_depth_value = config.MIN_DEPTH
            self.max_depth_value = config.MAX_DEPTH

        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=self.min_depth_value,
            high=self.max_depth_value,
            shape=(self.config.HEIGHT, self.config.WIDTH),
            dtype=np.float32,
        )

    def get_observation(self, sim_obs):
        obs = sim_obs.get(self.uuid, None)
        check_sim_obs(obs, self)

        obs = np.clip(obs, self.config.MIN_DEPTH, self.config.MAX_DEPTH)
        if self.config.NORMALIZE_DEPTH:
            # normalize depth observation to [0, 1]
            obs = (obs - self.config.MIN_DEPTH) / self.config.MAX_DEPTH

        return obs


class HabitatSimSemanticSensor(SemanticSensor):
    """Semantic sensor for habitat_sim
    """

    sim_sensor_type: habitat_sim.SensorType

    def __init__(self, config):
        self.sim_sensor_type = habitat_sim.SensorType.SEMANTIC
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.iinfo(np.uint32).min,
            high=np.iinfo(np.uint32).max,
            shape=(self.config.HEIGHT, self.config.WIDTH),
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

SIM_NAME_TO_ACTION = {v: k for k, v in SIM_ACTION_TO_NAME.items()}


class HabitatSim(habitat.Simulator):
    def __init__(self, config: Any) -> None:
        self.config = config.clone()
        agent_config = self._get_agent_config()

        sim_sensors = []
        for sensor_name in agent_config.SENSORS:
            sensor_cfg = getattr(self.config, sensor_name)
            is_valid_sensor = hasattr(
                habitat.sims.habitat_simulator, sensor_cfg.TYPE  # type: ignore
            )
            assert is_valid_sensor, "invalid sensor type {}".format(
                sensor_cfg.TYPE
            )
            sim_sensors.append(
                getattr(
                    habitat.sims.habitat_simulator,  # type: ignore
                    sensor_cfg.TYPE,
                )(sensor_cfg)
            )

        self.sensor_suite = SensorSuite(sim_sensors)
        self.sim_config = self.create_sim_config(self.sensor_suite)
        self._sim = habitat_sim.Simulator(self.sim_config)
        self.action_space = spaces.Discrete(
            len(self.sim_config.agents[0].action_space)
        )

        self.episode_active = False
        self._controls = SIM_ACTION_TO_NAME

    def create_sim_config(
        self, sensor_suite: SensorSuite
    ) -> habitat_sim.SimulatorConfiguration:
        sim_config = habitat_sim.SimulatorConfiguration()
        sim_config.scene.id = self.config.SCENE
        sim_config.gpu_device_id = self.config.HABITAT_SIM_V0.GPU_DEVICE_ID
        agent_config = habitat_sim.AgentConfiguration()
        overwrite_config(
            config_from=self._get_agent_config(), config_to=agent_config
        )

        sensor_specifications = []
        for sensor in sensor_suite.sensors.values():
            sim_sensor_cfg = habitat_sim.SensorSpec()
            sim_sensor_cfg.uuid = sensor.uuid
            sim_sensor_cfg.resolution = list(
                sensor.observation_space.shape[:2]
            )
            sim_sensor_cfg.parameters["hfov"] = str(sensor.config.HFOV)
            sim_sensor_cfg.position = sensor.config.POSITION
            # TODO(maksymets): Add configure method to Sensor API to avoid
            # accessing child attributes through parent interface
            sim_sensor_cfg.sensor_type = sensor.sim_sensor_type  # type: ignore
            sensor_specifications.append(sim_sensor_cfg)

        agent_config.sensor_specifications = sensor_specifications
        agent_config.action_space = {
            SimActions.LEFT.value: habitat_sim.ActionSpec(
                "lookLeft", {"amount": self.config.TURN_ANGLE}
            ),
            SimActions.RIGHT.value: habitat_sim.ActionSpec(
                "lookRight", {"amount": self.config.TURN_ANGLE}
            ),
            SimActions.FORWARD.value: habitat_sim.ActionSpec(
                "moveForward", {"amount": self.config.FORWARD_STEP_SIZE}
            ),
            SimActions.STOP.value: habitat_sim.ActionSpec("stop", {}),
        }
        sim_config.agents = [agent_config]
        return sim_config

    def _update_agents_state(self) -> bool:
        is_updated = False
        for agent_id, _ in enumerate(self.config.AGENTS):
            agent_cfg = self._get_agent_config(agent_id)
            if agent_cfg.IS_SET_START_STATE:
                self.set_agent_state(
                    agent_cfg.START_POSITION,
                    agent_cfg.START_ROTATION,
                    agent_id,
                )
                is_updated = True
        return is_updated

    def reset(self):
        sim_obs = self._sim.reset()
        if self._update_agents_state():
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
        is_same_scene = config.SCENE == self.config.SCENE
        self.config = config.clone()
        self.sim_config = self.create_sim_config(self.sensor_suite)
        if is_same_scene:
            self._sim.reconfigure(self.sim_config)
        else:
            self._sim.close()
            del self._sim
            self._sim = habitat_sim.Simulator(self.sim_config)

        self._update_agents_state()

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

    def _get_agent_config(self, agent_id: Optional[int] = None) -> Any:
        if agent_id is None:
            agent_id = self.config.DEFAULT_AGENT_ID
        agent_name = self.config.AGENTS[agent_id]
        agent_config = getattr(self.config, agent_name)
        return agent_config

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

    # TODO (maksymets): Remove check after simulator became stable
    def _check_agent_position(self, position, agent_id=0):
        if not np.allclose(position, self.agent_state(agent_id).position):
            logger.info("Agent state diverges from configured start position.")
