from home_robot.agent.motion.base import Planner
from home_robot.agent.motion.space import Space


def RRT(object):
    """Define RRT planning problem and parameters"""

    def __init__(self, space: Space, validate_fn, max_iter=1000):
        super(RRT, self).__init__(space, validate_fn)

    def plan(self, q0, qg):
        """plan from start to goal. creates a new tree"""
        raise NotImplementedError
