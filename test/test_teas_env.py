import os
import json

import numpy as np

import teas
from teas.config.experiments.esp_nav import esp_nav_cfg
from teas.tasks.nav.nav_task import NavigationEpisode
from teas.simulators.esp.esp_simulator import EspActions, ESP_ACTION_TO_NAME

MULTIHOUSE_RESOURCES_PATH = 'data/esp/multihouse-resources'
MULTIHOUSE_INITIALIZATIONS_PATH = 'data/esp/multihouse_initializations.json'
MULTIHOUSE_MAX_STEPS = 10


def test_vectorized_envs():
    assert os.path.exists(MULTIHOUSE_RESOURCES_PATH), \
        "Multihouse test data missing, " \
        "please download and place it in {}".format(MULTIHOUSE_RESOURCES_PATH)
    assert os.path.isfile(MULTIHOUSE_INITIALIZATIONS_PATH), \
        "Multhouse initialization points missing, " \
        "please download and place it in {}".format(
            MULTIHOUSE_INITIALIZATIONS_PATH)
    with open(MULTIHOUSE_INITIALIZATIONS_PATH, 'r') as f:
        multihouse_initializations = json.load(f)

    class TestDataset(teas.Dataset):
        def __init__(self, ind_house):
            house_id = sorted(os.listdir(MULTIHOUSE_RESOURCES_PATH))[ind_house]
            path = os.path.join(MULTIHOUSE_RESOURCES_PATH, house_id,
                                '{}.glb'.format(house_id))
            start_position = \
                multihouse_initializations[house_id]['start_position']
            start_rotation = \
                multihouse_initializations[house_id]['start_rotation']
            house_episode = NavigationEpisode(episode_id=str(i),
                                              scene_id=path,
                                              start_position=start_position,
                                              start_rotation=start_rotation,
                                              goals=[])
            self._episodes = [house_episode]

        @property
        def episodes(self):
            return self._episodes

    configs = []
    num_envs = len(os.listdir(MULTIHOUSE_RESOURCES_PATH))
    datasets = []
    for i in range(num_envs):
        datasets.append(TestDataset(i))

        config = esp_nav_cfg()
        config.task_name = 'Nav-v0'
        config.scene = datasets[-1].episodes[0].scene_id
        config.max_episode_steps = MULTIHOUSE_MAX_STEPS
        config.gpu_device_id = 0
        configs.append(config)

    envs = teas.VectorEnv(configs, datasets)
    envs.reset()
    dones = [False] * num_envs
    non_stop_actions = [k for k, v in ESP_ACTION_TO_NAME.items()
                        if v != EspActions.STOP.value]

    for i in range(2 * MULTIHOUSE_MAX_STEPS):
        observations, rewards, dones, infos = envs.step(
            np.random.choice(non_stop_actions, num_envs))
        assert len(observations) == num_envs
        assert len(rewards) == num_envs
        assert len(dones) == num_envs
        assert len(infos) == num_envs
        if (i + 1) % MULTIHOUSE_MAX_STEPS == 0:
            assert all(dones), "dones should be true after max_episode_steps"

    envs.close()
    assert all(dones), "dones should be true after max_episode_steps"


def test_teas_env():
    config = esp_nav_cfg()
    config.task_name = 'Nav-v0'
    assert os.path.exists(config.scene), \
        "ESP test data missing, please download and place it in data/esp/test/"
    env = teas.TeasEnv(config=config, dataset=None)
    env.episodes = [NavigationEpisode(
        episode_id="0",
        scene_id=config.scene,
        start_position=[03.00611, 0.072447, -2.67867],
        start_rotation=[0, 0.163276, 0, 0.98658],
        goals=[])]

    env.reset()
    done = False
    non_stop_actions = [k for k, v in ESP_ACTION_TO_NAME.items()
                        if v != EspActions.STOP.value]
    for _ in range(config.max_episode_steps):
        obs, rew, done, info = env.step(np.random.choice(non_stop_actions))

    # check for steps limit on environment
    assert done is True, "done should be true after max_episode_steps"
    env.close()
