import numpy as np


class Space(object):
    """class defining a region over which we can sample parameters"""

    def __init__(self, dof: int, mins, maxs):
        self.dof = dof
        self.mins = mins
        assert len(self.mins) == self.dof
        self.maxs = maxs
        assert len(self.maxs) == self.dof
        self.rngs = maxs - mins

    def sample_uniform(cls):
        return (np.random.random(self.dof) * self.rngs) + self.mins

    def extend(self, q0, q1, step_size=0.1):
        """extend towards another configuration in this space"""
