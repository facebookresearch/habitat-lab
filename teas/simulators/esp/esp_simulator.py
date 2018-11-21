from enum import Enum

import esp
import numpy as np
from gym import spaces, Space

import teas
from teas.core.simulator import RGBSensor

UUID_RGBSENSOR = 'rgb'
# ESP provides RGB as RGBD structure with 4 dimensions
RGBSENSOR_DIMENSION = 4


class EspRGBSensor(RGBSensor):
    def __init__(self, config, simulator):
        super().__init__()
        self._simulator = simulator
        self.observation_space = spaces.Box(low=0, high=255, shape=config.resolution + (RGBSENSOR_DIMENSION,),
                                            dtype=np.uint8)

    def observation(self):
        obs = self._simulator.cache.get(self.uuid)
        if obs is not None:
            obs = self._simulator.render()
            self._simulator.cache[self.uuid] = obs
        return obs


class EspActions(Enum):
    LEFT = 'look_left'
    RIGHT = 'look_right'
    FORWARD = 'move_forward'
    STOP = 'stop'


class EspSimulator(teas.Simulator):
    def __init__(self, config):
        # TODO(akadian): generalize this initialization. Use sensor configs,
        # move general parts to teas.Simulator, create predefined config for ESP
        self.esp_config = esp.SimulatorConfiguration()
        self.esp_config.scene.id = config.scene
        agent_config = esp.AgentConfiguration()
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
            EspActions.STOP.value: esp.ActionSpec('stop', {})
        }
        self.esp_config.agents = [agent_config]

        self._sim = esp.Simulator(self.esp_config)

        esp_sensors = []
        # TODO(akadian): Get rid of caching, use hooks into simulator for
        # sensor observations.
        self.cache = {}
        for s in config.sensors:
            assert hasattr(teas.simulators.esp, s), \
                'invalid sensor type {}'.format(s)
            esp_sensors.append(getattr(teas.simulators.esp, s)(config, self))
            self.cache[esp_sensors[-1].uuid] = None
        self.sensor_suite = teas.SensorSuite(esp_sensors)
        self.action_space = spaces.Discrete(len(agent_config.action_space))
        self.episode_active = False

        self._controls = {0: EspActions.LEFT.value,
                          1: EspActions.RIGHT.value,
                          2: EspActions.FORWARD.value,
                          3: EspActions.STOP.value}

    def reset(self):
        # TODO(akadian): remove caching once setup is finalized from ESP
        obs = self._sim.reset()
        self.cache[UUID_RGBSENSOR] = obs
        self.episode_active = True
        return self.sensor_suite.observations()

    def step(self, action):
        assert self.episode_active, \
            "episode is not active, environment not RESET or " \
            "STOP action called previously"
        sim_action = self._controls[action]
        if sim_action == EspActions.STOP.value:
            # TODO(akadian): Handle reward calculation on stop once pointnav
            # is integrated
            obs, rewards, done, info = None, None, True, None
            self.episode_active = False
            return obs, rewards, done, info
        obs, rewards, done, info = self._sim.step(sim_action)
        self.cache[UUID_RGBSENSOR] = obs
        observations = self.sensor_suite.observations()
        return observations, rewards, done, info

    def render(self):
        return self._sim.render()

    def seed(self, seed):
        self._sim.seed(seed)

    def reconfigure(self, *config):
        # TODO(akadian): Implement
        raise NotImplementedError

    def close(self):
        self._sim.close()

    def agent_state(self, agent_id=0):
        assert agent_id == 0, "No support of multi agent in {} yet.".format(self.__class__.__name__)
        return self._sim.last_state()

    def initialize_agent(self, position, rotation, agent_id=0):
        """
        :param position: numpy ndarray containing 3 entries for (x, y, z)
        :param rotation: numpy ndarray with 4 entries for (x, y, z, w) elements
                         of unit quaternion (versor) representing
                         agent 3D orientation,
                         ref: https://en.wikipedia.org/wiki/Versor
        :param agent_id: int identification of agent from multiagent setup
        """
        agent_state = esp.AgentState()
        agent_state.position = position
        agent_state.rotation = rotation

        assert agent_id == 0, "No support of multi agent in {} yet.".format(self.__class__.__name__)
        self._sim.initialize_agent(agent_id=agent_id,
                                   initial_state=agent_state)
