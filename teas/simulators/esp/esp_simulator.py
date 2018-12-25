from enum import Enum
from typing import List, Any, Dict, Optional

import esp
import numpy as np
from gym import spaces

import teas
from teas.core.logging import logger
from teas.core.simulator import Observation
from teas.core.simulator import RGBSensor

UUID_RGBSENSOR = 'rgb'
# ESP provides RGB as RGBD structure with 4 dimensions
RGBSENSOR_DIMENSION = 4


def overwrite_config(config_from: Dict, config_to) -> None:
    for attr, value in config_from.items():
        setattr(config_to, attr, value)


class EspRGBSensor(RGBSensor):
    def __init__(self, config, simulator):
        super().__init__()
        self._simulator = simulator
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=config.resolution + (
                                                RGBSENSOR_DIMENSION,),
                                            dtype=np.uint8)

    def get_observation(self):
        obs = self._simulator.cache.get(self.uuid)
        if obs is None:
            obs, _ = self._simulator.render()
            self._simulator.cache[self.uuid] = obs
        return obs


class EspActions(Enum):
    LEFT = 'look_left'
    RIGHT = 'look_right'
    FORWARD = 'move_forward'
    STOP = 'stop'


ESP_ACTION_TO_NAME = {
    0: EspActions.FORWARD.value,
    1: EspActions.LEFT.value,
    2: EspActions.RIGHT.value,
    3: EspActions.STOP.value
}


class EspSimulator(teas.Simulator):

    @staticmethod
    def create_esp_config(config: Any) -> esp.SimulatorConfiguration:
        # TODO(akadian): generalize this initialization. Use sensor configs,
        # move general parts to teas.Simulator, create predefined
        # config for ESP
        esp_config = esp.SimulatorConfiguration()
        # TODO(maksymets): use general notion of scene from Resource Manager
        esp_config.scene.id = config.scene
        agent_config = esp.AgentConfiguration()
        overwrite_config(config_from=config.agents[config.default_agent_id],
                         config_to=agent_config)
        color_sensor_config = esp.SensorSpec()
        color_sensor_config.resolution = config.resolution
        color_sensor_config.parameters['hfov'] = config.hfov
        color_sensor_config.position = config.sensor_position
        agent_config.sensor_specifications = [color_sensor_config]
        agent_config.action_space = {
            EspActions.LEFT.value: esp.ActionSpec(
                'lookLeft', {'amount': config.turn_angle}),
            EspActions.RIGHT.value: esp.ActionSpec(
                'lookRight', {'amount': config.turn_angle}),
            EspActions.FORWARD.value: esp.ActionSpec(
                'moveForward', {'amount': config.forward_step_size}),
            EspActions.STOP.value: esp.ActionSpec('stop', {}),
        }
        esp_config.agents = [agent_config]
        return esp_config

    def __init__(self, config) -> None:
        self.config = config
        self.esp_config = EspSimulator.create_esp_config(config)
        self.action_space = spaces.Discrete(
            len(self.esp_config.agents[0].action_space))
        self._sim = esp.Simulator(self.esp_config)

        esp_sensors = []
        # TODO(akadian): Get rid of caching, use hooks into simulator for
        # sensor observations.
        self.cache: Dict[str, Optional[Observation]] = {}
        for s in config.sensors:
            is_valid_sensor = hasattr(teas.simulators.esp, s)  # type: ignore
            assert is_valid_sensor, 'invalid sensor type {}'.format(s)
            esp_sensors.append(
                getattr(teas.simulators.esp, s)(config, self))  # type: ignore
            self.cache[esp_sensors[-1].uuid] = None
        self.sensor_suite = teas.SensorSuite(esp_sensors)
        self.episode_active = False
        self._controls = ESP_ACTION_TO_NAME

    def reset(self):
        # TODO(akadian): remove caching once setup is finalized from ESP
        obs = self._sim.reset()

        if hasattr(self.config, 'start_position') \
                and hasattr(self.config, 'start_rotation') \
                and self.config.start_position is not None \
                and self.config.start_rotation is not None:
            self.set_agent_state(self.config.start_position,
                                 self.config.start_rotation,
                                 self.config.default_agent_id)
            obs = self._sim.get_sensor_observations()
        self.cache[UUID_RGBSENSOR] = obs["rgba_camera"]
        self.episode_active = True
        return self.sensor_suite.get_observations(), False

    def step(self, action):
        assert self.episode_active, \
            "episode is not active, environment not RESET or " \
            "STOP action called previously"
        sim_action = self._controls[action]
        obs, done = Observation({}), False
        if sim_action == EspActions.STOP.value:
            # TODO(akadian): Handle reward calculation on stop once pointnav
            # is integrated
            done = True
            self.episode_active = False
            return obs, done
        obs = self._sim.step(sim_action)
        self.cache[UUID_RGBSENSOR] = obs["rgba_camera"]
        observations = self.sensor_suite.get_observations()
        return observations, done

    def render(self):
        obs, done = self._sim.get_sensor_observations(), False
        self.cache[UUID_RGBSENSOR] = obs["rgba_camera"]
        observations = self.sensor_suite.get_observations()
        return observations, done

    def seed(self, seed):
        self._sim.seed(seed)

    def reconfigure(self, config: Any) -> None:
        self.config = config
        self.esp_config = self.create_esp_config(config)
        self._sim.reconfigure(self.esp_config)
        if hasattr(config, 'start_position') \
                and hasattr(config, 'start_rotation') \
                and config.start_position is not None \
                and config.start_rotation is not None:
            self.initialize_agent(config.start_position, config.start_rotation,
                                  self.config.default_agent_id)

    def geodesic_distance(self, position_a, position_b):
        path = esp.ShortestPath()
        path.requested_start = position_a
        path.requested_end = position_b
        self._sim.pathfinder.find_path(path)
        return path.geodesic_distance

    def sample_navigable_point(self):
        r"""
        return: randomly sample a [x, y, z] point where the agent can be
        initialized.
        """
        return self._sim.pathfinder.get_random_navigable_point()

    def semantic_annotations(self):
        """
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
            self.__class__.__name__)
        state = esp.AgentState()
        self._sim.get_agent(agent_id).get_state(state)
        return state

    def set_agent_state(self, position: List[float] = None,
                        rotation: List[float] = None,
                        agent_id: int = 0) -> None:
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
        """
        :param position: numpy ndarray containing 3 entries for (x, y, z)
        :param rotation: numpy ndarray with 4 entries for (x, y, z, w) elements
        of unit quaternion (versor) representing agent 3D orientation,
        ref: https://en.wikipedia.org/wiki/Versor
        :param agent_id: int identification of agent from multiagent setup
        """
        agent_state = esp.AgentState()
        agent_state.position = position
        agent_state.rotation = rotation

        assert agent_id == 0, "No support of multi agent in {} yet.".format(
            self.__class__.__name__)
        self._sim.initialize_agent(agent_id=agent_id,
                                   initial_state=agent_state)
        self._check_agent_position(position, agent_id)

    # TODO (maksymets): Remove check after simulator became stable
    def _check_agent_position(self, position, agent_id=0):
        if not np.allclose(position, self.agent_state(agent_id).position):
            logger.info("Agent state diverges from configured start position.")
