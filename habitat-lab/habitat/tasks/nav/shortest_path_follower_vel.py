import math
import random
from typing import List, Optional, Union

import magnum as mn
import numpy as np
import quaternion

from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quat_to_rad, quaternion_rotate_vector


class ShortestPathFollowerv2:
    def __init__(
        self,
        sim: HabitatSim,
        goal_radius: float = 0.7,
    ):
        self._sim = sim
        self.goal_radius = goal_radius

    def get_yaw_from_matrix(self, rotation_matrix):
        return np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    def wrap_heading(self, heading):
        return (heading + np.pi) % (2 * np.pi) - np.pi

    def heading_error(
        self,
        agent_position: np.ndarray,
        position: np.ndarray,
        agent_heading: np.ndarray,
    ) -> float:
        heading_to_waypoint = np.arctan2(
            position[2] - agent_position[2], position[0] - agent_position[0]
        )
        agent_heading = self.wrap_heading(agent_heading)
        heading_error = self.wrap_heading(heading_to_waypoint - agent_heading)
        return heading_error

    def get_next_action(self, goal_pos):
        # Convert inputs to numpy arrays if they aren't already
        robot_pose = self._sim.articulated_agent.base_transformation
        robot_pos_YZX = robot_pose.translation
        robot_orientation = self.get_yaw_from_matrix(robot_pose.rotation())

        heading_err = self.heading_error(
            robot_pos_YZX, goal_pos, robot_orientation
        )

        if heading_err > np.deg2rad(30):
            return [0.0, -1.0]
        elif heading_err < -np.deg2rad(30):
            return [0.0, 1.0]
        else:
            robot_pos_XY = np.array([-robot_pos_YZX[2], robot_pos_YZX[0]])
            goal_pos_XY = np.array([-goal_pos[2], goal_pos[0]])
            distance_to_goal = np.linalg.norm(goal_pos_XY - robot_pos_XY)
            if distance_to_goal < self.goal_radius:
                return [0.0, 0.0]
            else:
                return [1.0, 0.0]
