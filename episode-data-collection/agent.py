import habitat_sim
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.geometry_utils import angle_between_quaternions, quaternion_from_coeff
from habitat_sim.utils.common import quat_to_angle_axis

from typing import Union, List, Optional

import numpy as np
import utils

class ImageNavShortestPathFollower(ShortestPathFollower):
    def __init__(
        self,
        sim: "HabitatSim",
        goal_radius: float,
        goal_pos: Union[List[float], np.ndarray],
        goal_rotation: Union[List[float], np.ndarray],  # or some orientation specification
        return_one_hot: bool = True,
        stop_on_error: bool = True,
        turn_angle: float = 15,
    ):
        super().__init__(sim, goal_radius, return_one_hot, stop_on_error)
        self.goal_pos = goal_pos
        self.goal_rotation = quaternion_from_coeff(goal_rotation)
        self.done = False
        self.turn_angle = np.deg2rad(turn_angle)

    def get_next_action(self, goal_pos=None) -> Optional[Union[int, np.ndarray]]:
        best_action = super().get_next_action(self.goal_pos)

        if self.done or best_action == HabitatSimActions.stop:
            self.done = True
            
            current_q = self._sim.get_agent_state().rotation
            goal_q = self.goal_rotation

            rel_q = goal_q * current_q.inverse()

            angle, axis = quat_to_angle_axis(rel_q)
            signed_yaw = angle if axis[1] >= 0 else -angle
            signed_yaw = (signed_yaw + np.pi) % (2 * np.pi) - np.pi

            if abs(signed_yaw) < self.turn_angle / 2:
                return HabitatSimActions.stop
            elif signed_yaw < 0:
                return HabitatSimActions.turn_right
            else:
                return HabitatSimActions.turn_left

        else:
            return best_action

