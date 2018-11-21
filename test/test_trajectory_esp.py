import json
import os

import numpy as np

from teas.config.experiments.esp_nav import get_default_config
from teas.simulators import make_simulator


def init_esp_sim():
    esp_nav_cfg = get_default_config()
    esp_nav_cfg.scene = os.environ['ESP_TEST_SCENE'] if 'ESP_TEST_SCENE' in os.environ else "data/esp/test/test.glb"
    esp_nav_cfg.freeze()
    assert os.path.exists(esp_nav_cfg.scene), "Please download ESP test data to data/esp/test/."
    return make_simulator('EspSimulator-v0', config=esp_nav_cfg)


def test_esp():
    with open('test/data/esp_trajectory_data.json', 'r') as f:
        test_trajectory = json.load(f)
    esp_simulator = init_esp_sim()

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

        assert esp_simulator.action_space.contains(action)
        obs, _, done, _ = esp_simulator.step(action)
        if i == len(test_trajectory['actions']) - 1:  # STOP action
            assert done is True
            assert esp_simulator.episode_active is False


if __name__ == '__main__':
    test_esp()
