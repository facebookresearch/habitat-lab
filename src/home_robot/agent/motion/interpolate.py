import scipy
from scipy.interpolate import CubicSpline


def interpolate(t, q):
    """using time and q generate a trajectory that we hope a robot could follow"""
