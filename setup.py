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

with open("LICENSE", encoding="utf-8") as f:
    license_text = f.read()

with open("requirements.txt", encoding="utf-8") as f:
    reqs = f.read()

DISTNAME = "habitat-lab"
DESCRIPTION = "Habitat Lab: a modular high-level library for end-to-end development in Embodied AI."
LONG_DESCRIPTION = readme
AUTHOR = "Facebook AI Research"
LICENSE = "MIT License"
REQUIREMENTS = reqs.strip().split("\n")
BASELINE_PATH = ["habitat_baselines", "habitat_baselines.*"]
DEFAULT_EXCLUSION = ["test", "examples"]
URL = "https://aihabitat.org/"
PROJECT_URLS = {
    "GitHub repo": "https://github.com/facebookresearch/habitat-lab/",
    "Bug Tracker": "https://github.com/facebookresearch/habitat-lab/issues",
}
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
        self.all = True  # None

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
        long_description_content_type="text/markdown",
        author=AUTHOR,
        license=LICENSE,
        setup_requires=["pytest-runner"],
        tests_require=["pytest-cov", "pytest-mock", "pytest"],
        include_package_data=True,
        cmdclass={"install": InstallCommand, "develop": DevelopCommand},
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
