# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

# TODO make this less brittle
sys.path = [os.path.join(os.path.dirname(__file__), "../")] + sys.path

import habitat  # isort:skip

import gym  # isort:skip
import typing  # isort:skip

# Override the typing annotation of _np_random of gym.Env since the type provided is
# "RandomNumberGenerator | None" and the "|" symbol for typing is not valid in Python <3.10
# Remove when upgrading to Python 3.10
gym.Env.__annotations__["_np_random"] = typing.Optional[
    gym.utils.seeding.RandomNumberGenerator
]


# Overrides the __all__ as that one pulls everything into the root module
# and doesn't expose any submodules
habitat.__all__ = [
    "config",
    "core",
    "gym",
    "articulated_agent_controllers",
    "articulated_agents",
    "datasets",
    "sims",
    "tasks",
    "utils",
]

habitat.core.__all__ = [
    "agent",
    "benchmark",
    "env",
    "embodied_task",
    "dataset",
    "simulator",
    "registry",
    "vector_env",
    "batch_rendering",
    "challenge",
    "environments",
    "logging",
    "utils",
]

habitat.datasets.__all__ = [
    "eqa",
    "image_nav",
    "object_nav",
    "pointnav",
    "rearrange",
    "vln",
    "utils",
    "registration",
]

habitat.datasets.rearrange.__all__ = [
    "navmesh_utils",
    "rearrange_dataset",
    "samplers",
]

habitat.datasets.object_nav.__all__ = [
    "object_nav_dataset",
]

habitat.datasets.pointnav.__all__ = [
    "pointnav_dataset",
]

habitat.sims.__all__ = [
    "habitat_simulator",
    "registration",
]

habitat.utils.visualizations.__all__ = [
    "fog_of_war",
    "maps",
    "utils",
]


PROJECT_TITLE = "Habitat"
PROJECT_SUBTITLE = "Lab Docs"
PROJECT_LOGO = "../../habitat-sim/docs/habitat.svg"
FAVICON = "../../habitat-sim/docs/habitat-blue.png"
MAIN_PROJECT_URL = "/"
INPUT_MODULES = [habitat]
INPUT_DOCS = ["docs.rst"]
INPUT_PAGES = [
    "pages/index.rst",
    "pages/quickstart.rst",
    "pages/habitat-lab-demo.rst",
    "pages/habitat-lab-tdmap-viz.rst",
    "pages/habitat2.rst",
    "pages/view-transform-warp.rst",
    "pages/metadata-taxonomy.rst",
]

PLUGINS = [
    "m.abbr",
    "m.code",
    "m.components",
    "m.dox",
    "m.gh",
    "m.htmlsanity",
    "m.images",
    "m.link",
    "m.math",
    "m.sphinx",
]

CLASS_INDEX_EXPAND_LEVELS = 2

PYBIND11_COMPATIBILITY = True
ATTRS_COMPATIBILITY = True

# Putting output into the sim repository so relative linking works the same
# way as on the website
OUTPUT = "../../habitat-sim/build/docs/habitat-lab/"

LINKS_NAVBAR1 = [
    (
        "Pages",
        "pages",
        [
            ("Quickstart", "quickstart"),
            ("Habitat Lab Demo", "habitat-lab-demo"),
            ("Habitat Lab TopdownMap Visualization", "habitat-lab-tdmap-viz"),
            ("Habitat 2.0 Overview", "habitat2"),
            ("View, Transform and Warp", "view-transform-warp"),
            ("'user_defined' Metadata Taxonomy", "metadata-taxonomy"),
        ],
    ),
    ("Classes", "classes", []),
]
LINKS_NAVBAR2 = [
    ("Sim Docs", "../habitat-sim/index.html", []),
]

FINE_PRINT = f"""
| {PROJECT_TITLE} {PROJECT_SUBTITLE}. Copyright © Meta Platforms.
| `Terms of Use </terms-of-use>`_ `Data Policy </data-policy>`_ `Cookie Policy </cookie-policy>`_
| Created with `m.css Python doc generator <https://mcss.mosra.cz/documentation/python/>`_."""

STYLESHEETS = [
    "https://fonts.googleapis.com/css?family=Source+Sans+Pro:400,400i,600,600i%7CSource+Code+Pro:400,400i,600",
    "../../habitat-sim/docs/theme.compiled.css",
]

M_SPHINX_INVENTORIES = [
    (
        "../../habitat-sim/docs/python.inv",
        "https://docs.python.org/3/",
        [],
        ["m-doc-external"],
    ),
    (
        "../../habitat-sim/docs/numpy.inv",
        "https://docs.scipy.org/doc/numpy/",
        [],
        ["m-doc-external"],
    ),
    (
        "../../habitat-sim/build/docs/habitat-sim/objects.inv",
        "../habitat-sim/",
        [],
        ["m-doc-external"],
    ),
]
M_SPHINX_INVENTORY_OUTPUT = "objects.inv"
M_SPHINX_PARSE_DOCSTRINGS = True

M_HTMLSANITY_SMART_QUOTES = True
# Will people hate me if I enable this?
# M_HTMLSANITY_HYPHENATION = True
