import sys

__version__ = '0.1.0'

try:
    __HABITAT_SETUP__  # type: ignore
except NameError:
    __HABITAT_SETUP__ = False

if __HABITAT_SETUP__:
    sys.stderr.write("Partial import of habitat during the install process.\n")
else:
    from habitat.core.dataset import Dataset
    from habitat.core.embodied_task import EmbodiedTask
    from habitat.core.simulator import SensorTypes, Sensor, SensorSuite, \
        Simulator
    from habitat.core.env import Env, RLEnv
    from habitat.core.vector_env import VectorEnv
    from habitat.core.logging import logger
    from habitat.datasets import make_dataset

    __all__ = ['Dataset', 'EmbodiedTask', 'Env', 'RLEnv',
               'Simulator', 'Sensor', 'logger', 'SensorTypes',
               'SensorSuite', 'VectorEnv', 'make_dataset']
