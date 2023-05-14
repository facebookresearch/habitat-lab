#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from habitat.utils.geometry_utils import (
    is_point_in_triangle,
    random_triangle_point,
)


def test_point_in_triangle_test():
    # contrived triangle test
    test_tri = (
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
    )
    test_pairs = [
        # corners
        (np.array([0.0, 0.0, 1.0]), True),
        (np.array([0.0, 0.99, 0.0]), True),
        (np.array([0, 0, 0]), True),
        # inside planar
        (np.array([0, 0.49, 0.49]), True),
        (np.array([0.0, 0.2, 0.2]), True),
        (np.array([0.0, 0.2, 0.4]), True),
        (np.array([0.0, 0.15, 0.3]), True),
        # outside but planar
        (np.array([0, 0, 1.01]), False),
        (np.array([0, 0, -0.01]), False),
        (np.array([0, 0.51, 0.51]), False),
        (np.array([0, -0.01, 0.51]), False),
        (np.array([0, -0.01, -0.01]), False),
        # inside non-planar
        (np.array([0.01, 0, 0]), False),
        (np.array([0.2, -0.01, 0.51]), False),
        (np.array([-0.2, -0.01, -0.01]), False),
        (np.array([0.1, 0.2, 0.2]), False),
        (np.array([-0.01, 0.2, 0.2]), False),
        # test epsilon padding around normal
        (np.array([1e-6, 0.1, 0.1]), False),
        (np.array([1e-7, 0.1, 0.1]), True),
    ]
    for test_pair in test_pairs:
        assert (
            is_point_in_triangle(
                test_pair[0], test_tri[0], test_tri[1], test_tri[2]
            )
            == test_pair[1]
        )


def test_random_triangle_point():
    # sample random points from random triangles, all should return True
    num_tris = 5
    num_samples = 10000
    for _tri in range(num_tris):
        v = [np.random.random(3) * 2 - np.ones(3) for _ in range(3)]
        sample_centroid = np.zeros(3)
        for _samp in range(num_samples):
            tri_point = random_triangle_point(v[0], v[1], v[2])
            assert is_point_in_triangle(tri_point, v[0], v[1], v[2])
            sample_centroid += tri_point
        # check uniformity of distribution by comparing sample centroid and triangle centroid
        sample_centroid /= num_samples
        true_centroid = (v[0] + v[1] + v[2]) / 3.0
        # print(np.linalg.norm(sample_centroid-true_centroid))
        # NOTE: need to be loose here because sample size is low
        assert np.allclose(sample_centroid, true_centroid, atol=0.01)
