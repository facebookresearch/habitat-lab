import json
import os

import numpy as np

from teas.config.experiments.esp_nav import esp_nav_cfg
from teas.simulators import make_simulator


def test_esp():
    current_directory = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(current_directory, 'data/esp_trajectory_data.json'),
              'r') as f:
        test_trajectory = json.load(f)
    
    assert 'ESP_TEST_SCENE' in os.environ, \
        'ESP_TEST_SCENE environment variable not defined'
    esp_nav_cfg.scene = os.environ['ESP_TEST_SCENE']
    esp_nav_cfg.freeze()
    
    esp_simulator = make_simulator('EspSimulator-v0', config=esp_nav_cfg)

    esp_simulator.reset()
    esp_simulator.initialize_agent(position=test_trajectory['positions'][0],
                                   rotation=test_trajectory['rotations'][0])
    
    for i, action in enumerate(test_trajectory['actions']):
        if i > 0:  # ignore first step as esp does not update agent until then
            state = esp_simulator.agent_state()
            assert np.allclose(
                np.array(test_trajectory['positions'][i], dtype=np.float32),
                state.position) is True
            assert np.allclose(
                np.array(test_trajectory['rotations'][i], dtype=np.float32),
                state.rotation) is True
        
        obs, _, done, _ = esp_simulator.step(action)
        if i == len(test_trajectory['actions']) - 1:  # STOP action
            assert done is True
            assert esp_simulator.episode_active is False


if __name__ == '__main__':
    test_esp()
