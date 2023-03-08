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


if __name__ == "__main__":
    setup(
        name="habitat-lab",
        install_requires=read("requirements.txt").strip().split("\n"),
        packages=find_packages(),
        version=get_package_version(),
        include_package_data=True,
        description="Habitat-Lab: a modular high-level library for end-to-end development in Embodied AI.",
        long_description=read("../README.md", encoding="utf8"),
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
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "Operating System :: MacOS",
            "Operating System :: Unix",
        ],
    )
