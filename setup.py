from setuptools import find_packages, setup

# TODO(akadian): Check if the below import works when dependencies are missing
from teas.version import VERSION

setup(
    name='teas',
    packages=find_packages(),
    version=VERSION,
    description="The Embodied Agent Suite (TEAS): A suite for embodied agent "
                "tasks and benchmarks.",
    author='fair',
    license='',
)
