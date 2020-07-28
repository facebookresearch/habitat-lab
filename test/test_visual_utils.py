#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from habitat.utils.visualizations.utils import observations_to_image


def test_observations_to_image():
    observations = {
        "rgb": np.random.rand(200, 400, 3),
        "depth": np.random.rand(200, 400, 1),
    }
    info = {
        "collisions": {"is_collision": True},
        "top_down_map": {
            "map": np.random.randint(low=0, high=255, size=(300, 300)),
            "fog_of_war_mask": np.random.randint(
                low=0, high=1, size=(300, 300)
            ),
            "agent_map_coord": (10, 10),
            "agent_angle": np.random.random(),
        },
    }
    image = observations_to_image(observations, info)
    assert image.shape == (
        200,
        1000,
        3,
    ), "Resulted image resolution doesn't match."
