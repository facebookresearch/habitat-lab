# The ONLY two lines you need to add to start importing Habitat 2.0 Gym environments.
import gym

# flake8: noqa
import habitat.utils.gym_definitions

import pytest

@pytest.mark.parametrize(
    "name",
    [
        "HabitatPick-v0",
        "HabitatPlace-v0",
        "HabitatCloseCab-v0",
        "HabitatCloseFridge-v0",
        "HabitatOpenCab-v0",
        "HabitatOpenFridge-v0",
        "HabitatNavToObj-v0",
        "HabitatReachState-v0",
        "HabitatNavPick-v0",
        "HabitatNavPickNavPlace-v0",
        "HabitatTidyHouse-v0",
        "HabitatPrepareGroceries-v0",
        "HabitatSetTable-v0",
        "HabitatRenderPick-v0",
        "HabitatRenderPlace-v0",
        "HabitatRenderCloseCab-v0",
        "HabitatRenderCloseFridge-v0",
        "HabitatRenderOpenCab-v0",
        "HabitatRenderOpenFridge-v0",
        "HabitatRenderNavToObj-v0",
        "HabitatRenderReachState-v0",
        "HabitatRenderNavPick-v0",
        "HabitatRenderNavPickNavPlace-v0",
        "HabitatRenderTidyHouse-v0",
        "HabitatRenderPrepareGroceries-v0",
        "HabitatRenderSetTable-v0",
    ],
)
def test_gym(name):
    env = gym.make(name)
    done = False
    cum_rew = 0.0
    env.reset()
    while not done:
        _, reward, done, _ = env.step(env.action_space.sample())
        cum_rew += reward
    print(cum_rew)
    env.close()