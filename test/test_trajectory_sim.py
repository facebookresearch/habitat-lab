import json
import os

import numpy as np
from habitat.sims import make_sim
from habitat.config.default import cfg


def init_sim():
    config = cfg()
    config.SCENE = "data/habitat-sim/test/test.glb"
    assert os.path.exists(
        config.SCENE
    ), "Please download Habitat-Sim test data to data/habitat-sim/test/."
    return make_sim(config.SIMULATOR.TYPE, config=config.SIMULATOR)


def test_sim():
    with open("test/data/habitat-sim_trajectory_data.json", "r") as f:
        test_trajectory = json.load(f)
    sim = init_sim()

    sim.reset()
    sim.set_agent_state(
        position=test_trajectory["positions"][0],
        rotation=test_trajectory["rotations"][0],
    )

    for i, action in enumerate(test_trajectory["actions"]):
        if i > 0:  # ignore first step as habitat-sim doesn't update
            # agent until then
            state = sim.agent_state()
            assert (
                np.allclose(
                    np.array(
                        test_trajectory["positions"][i], dtype=np.float32
                    ),
                    state.position,
                )
                is True
            ), "mismatch in position " "at step {}".format(i)
            assert (
                np.allclose(
                    np.array(
                        test_trajectory["rotations"][i], dtype=np.float32
                    ),
                    state.rotation,
                )
                is True
            ), "mismatch in rotation " "at step {}".format(i)
        assert sim.action_space.contains(action)

        obs = sim.step(action)
        if i == len(test_trajectory["actions"]) - 1:  # STOP action
            assert sim.episode_active is False

    sim.close()
