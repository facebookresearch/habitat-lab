import json
import os

import numpy as np
from habitat.config.experiments.nav import sim_nav_cfg
from habitat.sims import make_sim


def init_sim():
    config = sim_nav_cfg()
    config.scene = os.environ.get('ESP_TEST_SCENE', 'data/esp/test/test.glb')
    assert os.path.exists(config.scene), \
        "Please download ESP test data to data/esp/test/."
    return make_sim('Sim-v0', config=config)


def test_sim():
    with open('test/data/esp_trajectory_data.json', 'r') as f:
        test_trajectory = json.load(f)
    sim = init_sim()

    sim.reset()
    sim.initialize_agent(position=test_trajectory['positions'][0],
                         rotation=test_trajectory['rotations'][0])

    for i, action in enumerate(test_trajectory['actions']):
        if i > 0:  # ignore first step as esp does not update agent until then
            state = sim.agent_state()
            assert np.allclose(
                np.array(test_trajectory['positions'][i], dtype=np.float32),
                state.position) is True, "mismatch in position " \
                                         "at step {}".format(i)
            assert np.allclose(
                np.array(test_trajectory['rotations'][i], dtype=np.float32),
                state.rotation) is True, "mismatch in rotation " \
                                         "at step {}".format(i)
        assert sim.action_space.contains(action)

        obs, done = sim.step(action)
        if i == len(test_trajectory['actions']) - 1:  # STOP action
            assert done is True
            assert sim.episode_active is False

    sim.close()
