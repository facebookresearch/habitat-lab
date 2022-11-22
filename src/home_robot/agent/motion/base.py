# Copyright Meta 2022

from home_robot.agent.motion.robot import Robot
from home_robot.agent.motion.space import Space

"""
This just defines the standard interface for a motion planner
"""


class Planner(object):
    """planner base class"""

    # def __init__(self, space: Space, validate_fn):
    def __init__(self, robot: Robot):
        self.robot = robot
        # self.Space = space
        # self.validate = validate_fn

    def plan(self, q0, qg):
        """returns a trajectory"""
        raise NotImplementedError
