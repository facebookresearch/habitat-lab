import builtins

from setuptools import find_packages, setup

DISTNAME = "habitat"
DESCRIPTION = "habitat: a suite for embodied agent tasks and benchmarks"
AUTHOR = "fair"
LICENSE = "TODO"  # TODO(akadian): add license
REQUIREMENTS = ["gym==0.10.9", "h5py", "numpy", "yacs"]

# import restricted version of habitat to get __version__
builtins.__HABITAT_SETUP__ = True  # type: ignore
import habitat  # noqa

VERSION = habitat.__version__

setup(
    name=DISTNAME,
    install_requires=REQUIREMENTS,
    packages=find_packages(),
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    license=LICENSE,
)
