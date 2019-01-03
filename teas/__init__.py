import sys

__version__ = '0.1.0'

try:
    __TEAS_SETUP__  # type: ignore
except NameError:
    __TEAS_SETUP__ = False

if __TEAS_SETUP__:
    sys.stderr.write("Partial import of teas during the install process.\n")
else:
    from teas.core.dataset import Dataset
    from teas.core.embodied_task import EmbodiedTask
    from teas.core.simulator import SensorTypes, Sensor, SensorSuite, Simulator
    from teas.core.env import TeasEnv
    from teas.core.vector_env import VectorEnv
    from teas.core.logging import logger
    from teas.tasks import make_task

    __all__ = ['Dataset', 'EmbodiedTask', 'TeasEnv', 'Simulator',
               'Sensor', 'SensorTypes', 'SensorSuite', 'VectorEnv',
               'make_task']
