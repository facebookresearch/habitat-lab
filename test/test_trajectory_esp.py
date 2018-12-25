import json
import os

import numpy as np

from teas.config.experiments.esp_nav import esp_nav_cfg
from teas.simulators import make_simulator


def init_esp_sim():
    config = esp_nav_cfg()
    config.scene = os.environ.get('ESP_TEST_SCENE', 'data/esp/test/test.glb')
    assert os.path.exists(config.scene), \
        "Please download ESP test data to data/esp/test/."
    return make_simulator('EspSimulator-v0', config=config)


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
                state.position) is True, "mismatch in position " \
                                         "at step {}".format(i)
            assert np.allclose(
                np.array(test_trajectory['rotations'][i], dtype=np.float32),
                state.rotation) is True, "mismatch in rotation " \
                                         "at step {}".format(i)
        assert esp_simulator.action_space.contains(action)

        obs, done = esp_simulator.step(action)
        if i == len(test_trajectory['actions']) - 1:  # STOP action
            assert done is True
            assert esp_simulator.episode_active is False

    esp_simulator.close()


if __name__ == '__main__':
    test_esp()
