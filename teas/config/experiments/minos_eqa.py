import os

from yacs.config import CfgNode


def minos_eqa_cfg():
    config = CfgNode()
    assert 'MINOS_EQA_DATA' in os.environ, \
        'environment variable MINOS_EQA_DATA not defined'
    config.dataset = CfgNode()
    config.dataset.data_path = os.environ['MINOS_EQA_DATA']
    config.dataset.split = 'train'

    config.env = CfgNode()
    config.env.frameskip = 1
    config.env.angle = 5
    config.env.seed = 100
    config.env.framerate = 24
    config.env.sensors = ['MinosRGBSensor']
    config.env.simulator = 'MinosSimulator-v0'
    config.env.max_episode_seconds = 100000
    config.env.max_episode_steps = 1000
    return config
