import numpy as np


class FrankaPanda(object):
    """contains information about the franka robot"""

    def __init__(self):
        # Distance from ee frame
        grasp_offset = np.eye(4)
        grasp_offset[2, 3] = 0.22
        self.grasp_offset = grasp_offset

    def apply_grasp_offset(self, ee_pose):
        return ee_pose @ self.grasp_offset
