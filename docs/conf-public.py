# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# Inherit everything from the local config
from conf import *  # isort:skip

OUTPUT = "../../habitat-sim/build/docs-public/habitat-lab/"

SEARCH_DOWNLOAD_BINARY = "searchdata-v1.bin"
SEARCH_BASE_URL = "https://aihabitat.org/docs/habitat-lab/"
SEARCH_EXTERNAL_URL = "https://google.com/search?q=site:aihabitat.org+{query}"

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
        "../../habitat-sim/build/docs-public/habitat-sim/objects.inv",
        "../habitat-sim/",
        [],
        ["m-doc-external"],
    ),
]
