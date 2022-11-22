#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import sys

import setuptools
from setuptools.command.develop import develop as DefaultDevelopCommand
from setuptools.command.install import install as DefaultInstallCommand

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "habitat"))
from version import VERSION  # isort:skip noqa


with open("../README.md", encoding="utf8") as f:
    readme = f.read()


with open("requirements.txt") as f:
    reqs = f.read()

DISTNAME = "habitat-lab"
DESCRIPTION = "Habitat Lab: a modular high-level library for end-to-end development in Embodied AI."
LONG_DESCRIPTION = readme
AUTHOR = "Facebook AI Research"
LICENSE = "MIT License"
REQUIREMENTS = reqs.strip().split("\n")
DEFAULT_EXCLUSION = ["tests"]
URL = "https://aihabitat.org/"
PROJECT_URLS = {
    "GitHub repo": "https://github.com/facebookresearch/habitat-lab/",
    "Bug Tracker": "https://github.com/facebookresearch/habitat-lab/issues",
}

if __name__ == "__main__":
    # package data are the files and configurations included in the package
    package_data = [
        x[8:] for x in glob.glob("habitat/config/**/*.yaml", recursive=True)
    ] + ["utils/visualizations/assets/**/*.png"]
    setuptools.setup(
        name=DISTNAME,
        install_requires=REQUIREMENTS,
        packages=setuptools.find_packages(exclude=DEFAULT_EXCLUSION),
        package_data={"habitat": package_data},
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        author=AUTHOR,
        license=LICENSE,
        setup_requires=["pytest-runner"],
        tests_require=[
            "pytest-cov",
            "pytest-mock",
            "pytest",
            "pybullet==3.0.4",
            "mock",
        ],
        include_package_data=True,
        cmdclass={
            "install": DefaultInstallCommand,
            "develop": DefaultDevelopCommand,
        },
        url=URL,
        project_urls=PROJECT_URLS,
        classifiers=[
            "Intended Audience :: Science/Research",
            "Development Status :: 5 - Production/Stable",
            "License :: OSI Approved :: MIT License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "Operating System :: MacOS",
            "Operating System :: Unix",
        ],
    )
