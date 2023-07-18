#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob

from setuptools import find_packages, setup


def read(file_path, *args, **kwargs):
    with open(file_path, *args, **kwargs) as f:
        content = f.read()
    return content


def collect_requirements():
    # collect requirements.txt file in all subdirectories
    requirements = set()
    for file_path in glob.glob(
        "habitat_baselines/**/requirements.txt", recursive=True
    ):
        content = read(file_path)
        requirements.update(content.strip().split("\n"))

    return list(requirements)


def get_package_version():
    import os.path as osp
    import sys

    sys.path.insert(0, osp.join(osp.dirname(__file__), "habitat_baselines"))
    from version import VERSION

    return VERSION


if __name__ == "__main__":
    setup(
        name="habitat-baselines",
        install_requires=collect_requirements(),
        packages=find_packages(),
        version=get_package_version(),
        include_package_data=True,
        description="Habitat-Baselines: Embodied AI baselines.",
        long_description=read("README.md", encoding="utf8"),
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
        entry_points={
            "console_scripts": [
                "habitat-baselines=habitat_baselines.run:main",
            ],
        },
    )
