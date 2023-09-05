# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np


EPSILON = np.finfo(float).eps

EYE_R = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], float)

EYE_T = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    float,
)

ZERO_P = np.array([0.0, 0.0, 0.0], float)

ZERO_R = np.zeros((3, 3))


def eye_T():
    return EYE_T.copy()


def eye_R():
    return EYE_R.copy()


def zero_p():
    return ZERO_P.copy()


def zero_R():
    return ZERO_R.copy()
