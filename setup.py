#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os.path
import sys

import setuptools
from setuptools.command.develop import develop as DefaultDevelopCommand
from setuptools.command.install import install as DefaultInstallCommand

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "habitat"))
from version import VERSION  # isort:skip noqa


with open("README.md", encoding="utf8") as f:
    readme = f.read()

with open("LICENSE") as f:
    license_text = f.read()

with open("requirements.txt") as f:
    reqs = f.read()

DISTNAME = "habitat"
DESCRIPTION = "habitat: a suite for embodied agent tasks and benchmarks"
LONG_DESCRIPTION = readme
AUTHOR = "Facebook AI Research"
LICENSE = license_text
REQUIREMENTS = reqs.strip().split("\n")
BASELINE_PATH = ["habitat_baselines", "habitat_baselines.*"]
DEFAULT_EXCLUSION = ["test", "examples"]
FULL_REQUIREMENTS = set()
# collect requirements.txt file in all subdirectories
for file_name in ["requirements.txt"] + glob.glob(
    "habitat_baselines/**/requirements.txt", recursive=True
):
    with open(file_name) as f:
        reqs = f.read()
        FULL_REQUIREMENTS.update(reqs.strip().split("\n"))


class OptionedCommand:
    r"""Generic Command class that takes extra user options and modifies
    arguments in setuptools.setup() accordingly.
    Though OptionedCommand inherits directly from object, it assumes
    inheritance from DefaultDevelopCommand or DefaultInstallCommand, as it
    overrides methods from those two classes.
    """

    user_options = [("all", None, "include habitat_baselines in installation")]

    def initialize_options(self):
        super().initialize_options()
        self.all = None

    def run(self):
        if not self.all:  # install core only
            DEFAULT_EXCLUSION.extend(BASELINE_PATH)
            self.distribution.packages = setuptools.find_packages(
                exclude=DEFAULT_EXCLUSION
            )
            # self.distribution accesses arguments of setup() in main()
        else:  # install all except test and examples
            self.distribution.install_requires = FULL_REQUIREMENTS
        super().run()


class InstallCommand(OptionedCommand, DefaultInstallCommand):
    user_options = (
        getattr(DefaultInstallCommand, "user_options", [])
        + OptionedCommand.user_options
    )


class DevelopCommand(OptionedCommand, DefaultDevelopCommand):
    user_options = (
        getattr(DefaultDevelopCommand, "user_options", [])
        + OptionedCommand.user_options
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
        tests_require=[
            "pytest-cov",
            "pytest-mock",
            "pytest",
            "pybullet==3.0.4",
            "mock",
        ],
        include_package_data=True,
        cmdclass={"install": InstallCommand, "develop": DevelopCommand},
    )
