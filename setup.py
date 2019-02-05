#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import builtins

from setuptools import find_packages, setup


with open('README.md', encoding="utf8") as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    reqs = f.read()

DISTNAME = "habitat"
DESCRIPTION = "habitat: a suite for embodied agent tasks and benchmarks"
LONG_DESCRIPTION = readme
AUTHOR = "Facebook AI Research"
LICENSE = license
REQUIREMENTS = reqs.strip().split('\n'),

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
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    license=LICENSE,
)
