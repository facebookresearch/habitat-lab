import esp
import numpy as np
from gym import spaces, Space

import teas
from teas.core.simulator import RGBSensor

UUID_RGBSENSOR = 'rgb'


class EspRGBSensor(RGBSensor):
    def __init__(self, simulator):
        super().__init__()
        self._simulator = simulator
        self.observation_space = Space(
            shape=self._simulator.esp_config['resolution'],
            dtype=np.uint8)
    
    def observation(self):
        obs = self._simulator.cache.get(self.uuid)
        if obs is not None:
            obs = self._simulator.render()
            self._simulator.cache[self.uuid] = obs
        return obs


class EspSimulator(teas.Simulator):
    def __init__(self, config):
        self.esp_config = {'resolution': config.resolution,
                           'scene': config.scene}
        self._sim = esp.Simulator(self.esp_config)
        esp_sensors = []
        # TODO(akadian): Get rid of caching, use hooks into simulator for
        # sensor observations.
        self.cache = {}
        for s in config.sensors:
            assert hasattr(teas.simulators.esp, s), \
                'invalid sensor type {}'.format(s)
            esp_sensors.append(getattr(teas.simulators.esp, s)(self))
            self.cache[esp_sensors[-1].uuid] = None
        self.sensor_suite = teas.SensorSuite(esp_sensors)
        self.action_space = spaces.Discrete(len(self._sim.action_space))
        self._controls = self._sim.action_space
    
    def reset(self):
        # TODO(akadian): remove caching once setup is finalized from ESP
        obs = self._sim.reset()
        self.cache[UUID_RGBSENSOR] = obs
        return self.sensor_suite.observations()
    
    def step(self, action):
        sim_action = self._controls[action]
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
