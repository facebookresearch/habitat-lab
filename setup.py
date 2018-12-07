import builtins

from setuptools import find_packages, setup

DISTNAME = 'teas'
DESCRIPTION = "The Embodied Agent Suite (TEAS): A suite for embodied agent " \
              "tasks and benchmarks"
AUTHOR = 'fair'
LICENSE = 'TODO'  # TODO(akadian): add license
REQUIREMENTS = ['gym==0.10.9', 'numpy', 'yacs']

# import restricted version of teas to get __version__
builtins.__TEAS_SETUP__ = True
import teas

VERSION = teas.__version__

setup(
    name=DISTNAME,
    install_requires=REQUIREMENTS,
    packages=find_packages(),
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    license=LICENSE,
)
