#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from setuptools import find_packages, setup


def read(file_path, *args, **kwargs):
    with open(file_path, *args, **kwargs) as f:
        content = f.read()
    return content


def get_package_version():
    import os.path as osp
    import sys

    sys.path.insert(0, osp.join(osp.dirname(__file__), "habitat"))
    from version import VERSION

    return VERSION


def get_long_description():
    return """
[![codecov](https://codecov.io/gh/facebookresearch/habitat-lab/branch/main/graph/badge.svg)](https://codecov.io/gh/facebookresearch/habitat-lab)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/facebookresearch/habitat-lab/blob/main/LICENSE)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/facebookresearch/habitat-lab)](https://github.com/facebookresearch/habitat-lab/releases/latest)
[![Supports Habitat_Sim](https://img.shields.io/static/v1?label=supports&message=Habitat%20Sim&color=informational&link=https://github.com/facebookresearch/habitat-sim)](https://github.com/facebookresearch/habitat-sim)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://timothycrosley.github.io/isort/)
[![Twitter Follow](https://img.shields.io/twitter/follow/ai_habitat?style=social)](https://twitter.com/ai_habitat)

Habitat-Lab
==============================

Habitat-Lab is a modular high-level library for end-to-end development in embodied AI --
defining embodied AI tasks (e.g. navigation, rearrangement, instruction following, question answering),
configuring embodied agents (physical form, sensors, capabilities), training these agents (via imitation
or reinforcement learning, or no learning at all as in SensePlanAct pipelines), and benchmarking their
performance on the defined tasks using standard metrics.

Habitat-Lab uses [`Habitat-Sim`](https://github.com/facebookresearch/habitat-sim) as the core simulator.
For documentation refer [here](https://aihabitat.org/docs/habitat-lab/).

[![Habitat Demo](https://img.shields.io/static/v1?label=WebGL&message=Try%20AI%20Habitat%20In%20Your%20Browser%20&color=blue&logo=webgl&labelColor=%23990000&style=for-the-badge&link=https://aihabitat.org/demo)](https://aihabitat.org/demo)
"""


if __name__ == "__main__":
    setup(
        name="habitat-lab",
        install_requires=read("requirements.txt").strip().split("\n"),
        packages=find_packages(),
        version=get_package_version(),
        include_package_data=True,
        description="Habitat-Lab: a modular high-level library for end-to-end development in Embodied AI.",
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        author="Meta AI Research",
        license="MIT License",
        url="https://aihabitat.org",
        project_urls={
            "GitHub repo": "https://github.com/facebookresearch/habitat-lab/",
            "Bug Tracker": "https://github.com/facebookresearch/habitat-lab/issues",
        },
        classifiers=[
            "Intended Audience :: Science/Research",
            "Development Status :: 5 - Production/Stable",
            "License :: OSI Approved :: MIT License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "Operating System :: MacOS",
            "Operating System :: Unix",
        ],
    )
