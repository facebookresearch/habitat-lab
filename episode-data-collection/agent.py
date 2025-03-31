import habitat_sim
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.geometry_utils import angle_between_quaternions, quaternion_from_coeff

from typing import Union, List, Optional

import numpy as np
import utils

class ImageNavShortestPathFollower(ShortestPathFollower):
    def __init__(
        self,
        sim: "HabitatSim",
        goal_radius: float,
        goal_rotation: Union[List[float], np.ndarray],  # or some orientation specification
        return_one_hot: bool = True,
        stop_on_error: bool = True,
    ):
        super().__init__(sim, goal_radius, return_one_hot, stop_on_error)
        self.goal_rotation = quaternion_from_coeff(goal_rotation)
        self.done = False

    def get_next_action(
        self, goal_pos: Union[List[float], np.ndarray]
    ) -> Optional[Union[int, np.ndarray]]:
        best_action = super().get_next_action(goal_pos)

        if self.done or best_action == HabitatSimActions.stop:
            self.done = True
            
            current_quat = self._sim.get_agent_state().rotation
            yaw_error = (angle_between_quaternions(current_quat, self.goal_rotation) + np.pi) % (2 * np.pi) - np.pi
            if np.abs(yaw_error) < 0.3:
                return HabitatSimActions.stop
            elif yaw_error > 0:
                return HabitatSimActions.turn_left
            else:
                return HabitatSimActions.turn_right
        else:
            return best_action

