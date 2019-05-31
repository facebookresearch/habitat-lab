#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path
import sys
import glob
import setuptools
from setuptools.command.develop import develop
from setuptools.command.install import install

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "habitat"))
from version import VERSION  # noqa

with open("README.md", encoding="utf8") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

with open("requirements.txt") as f:
    reqs = f.read()

DISTNAME = "habitat"
DESCRIPTION = "habitat: a suite for embodied agent tasks and benchmarks"
LONG_DESCRIPTION = readme
AUTHOR = "Facebook AI Research"
LICENSE = license
REQUIREMENTS = (reqs.strip().split("\n"),)
DEFAULT_EXCLUSION = ["test", "examples"]
FULL_REQUIREMENTS = []
for file_name in glob.glob("**/*requirements.txt", recursive=True):
    with open(file_name) as f:
        reqs = f.read()
        FULL_REQUIREMENTS.extend(reqs.strip().split("\n"))


class OptionedCommand(object):
    user_options = [
        ("include_baseline", None, "include habitat_baselines in installation")
    ]

    def initialize_options(self):
        super().initialize_options(self)
        self.include_baseline = None

    def finalize_options(self):
        super().finalize_options(self)

    def run(self):
        if not self.include_baseline:
            DEFAULT_EXCLUSION.append("baselines")
        else:
            REQUIREMENTS[:] = FULL_REQUIREMENTS
        super().run(self)


class InstallCommand(OptionedCommand, install):
    user_options = (
        getattr(install, "user_options", []) + OptionedCommand.user_options
    )


class DevelopCommand(OptionedCommand, develop):
    user_options = (
        getattr(develop, "user_options", []) + OptionedCommand.user_options
    )


if __name__ == "__main__":
    setuptools.setup(
        name=DISTNAME,
        install_requires=REQUIREMENTS,
        packages=setuptools.find_packages(exclude=DEFAULT_EXCLUSION),
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author=AUTHOR,
        license=LICENSE,
        setup_requires=["pytest-runner"],
        tests_require=["pytest"],
        include_package_data=True,
        cmdclass={"install": InstallCommand, "develop": DevelopCommand},
    )
