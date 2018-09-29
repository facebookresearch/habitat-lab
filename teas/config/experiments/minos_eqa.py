import os

from yacs.config import CfgNode

_minos_eqa_c = CfgNode()
assert 'MINOS_EQA_DATA' in os.environ, 'environment variable MINOS_EQA_DATA ' \
                                       'not defined'
_minos_eqa_c.dataset = CfgNode()
_minos_eqa_c.dataset.data_path = os.environ['MINOS_EQA_DATA']
_minos_eqa_c.dataset.split = 'train'

_minos_eqa_c.env = CfgNode()
_minos_eqa_c.env.frameskip = 1
_minos_eqa_c.env.angle = 5
_minos_eqa_c.env.seed = 100
_minos_eqa_c.env.framerate = 24
_minos_eqa_c.env.sensors = ['MinosRGBSensor']
_minos_eqa_c.env.simulator = 'MinosSimulator-v0'

minos_eqa_cfg = _minos_eqa_c
