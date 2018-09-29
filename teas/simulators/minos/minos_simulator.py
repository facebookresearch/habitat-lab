import numpy as np
from gym import spaces, Space
from minos.lib import common
from minos.lib.Simulator import Simulator

import teas
from teas.core.simulator import RGBSensor
from teas.simulators.minos.utils import minos_args


class MinosRGBSensor(RGBSensor):
    def __init__(self, simulator):
        super().__init__()
        self.observation_space = Space(shape=(256, 256, 4),
                                       dtype=np.uint8)
        self._simulator = simulator
    
    def observation(self):
        sim_obs = self._simulator.get_last_observation().get('observation').get(
            'sensors')['color']['data']
        return sim_obs


class MinosSimulator(teas.Simulator):
    def __init__(self, config):
        minos_params = minos_args(config)
        self._last_state = None
        self._sim = Simulator(minos_params)
        self.angle = minos_params['angle']
        self.frameskip = minos_params['frameskip']
        self.dataset = minos_params['scene']['dataset']
        self.available_controls = minos_params.get('available_controls',
                                                   ['turnLeft', 'turnRight',
                                                    'forwards'])
        minos_sensors = []
        for s in minos_params['sensors']:
            assert hasattr(teas.simulators.minos, s), \
                'invalid sensor type {}'.format(s)
            minos_sensors.append(getattr(teas.simulators.minos, s)(self._sim))
        self.sensor_suite = teas.SensorSuite(minos_sensors)
        self.action_space = spaces.Discrete(len(
            self.available_controls))
        self._sim.start()
        common.attach_exit_handler(self._sim)
        self.viewer = None
    
    @property
    def last_state(self):
        return self._last_state
    
    def set_scene(self, scene_id):
        self._sim.set_scene(self.dataset + '.' + scene_id)
        episode_info = self._sim.start()
        return episode_info
    
    def reset(self):
        res = self._sim.reset()
        obs = None
        if res is not False:
            obs = self._sim.get_last_observation()
        return obs
    
    def step(self, action):
        # TODO(akadian): In the default setting of MINOS done and rewards are
        #  returned as None, this should be adapted
        # according to the task the environment is being used for.
        sim_action = {'name': self.available_controls[action], 'strength': 1,
                      'angle': self.angle}
        state = self._sim.step(sim_action, self.frameskip)
        self._last_state = state
        observation = self.sensor_suite.observations()
        if state is None:
            return None, None, None, None
        rewards = state.get('rewards')
        done = state.get('terminals')
        info = state.get('info')
        return observation, rewards, done, info
    
    def seed(self, seed):
        self._sim.seed(seed)
    
    def render(self, mode, close):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None  # If we don't None out this reference
                # pyglet becomes unhappy
            return
        if self.last_state is not None:
            img = self.last_state['observation']['sensors']['color'][
                'data']
            if len(img.shape) == 2:  # assume gray
                img = np.dstack([img, img, img])
            else:  # assume rgba
                img = img[:, :, :-1]
            img = img.reshape((img.shape[1], img.shape[0], img.shape[2]))
            if mode == 'human':
                from gym.envs.classic_control import rendering
                if self.viewer is None:
                    if self.viewer is None:
                        self.viewer = rendering.SimpleImageViewer()
                self.viewer.imshow(img)
            elif mode == 'rgb_array':
                return img
            else:
                raise NotImplemented
    
    def reconfigure(self, house_id):
        self.set_scene(house_id)
    
    def close(self):
        if self._sim is not None:
            self._sim.kill()
