from typing import Optional

import numpy as np
from home_robot.control.goto_controller import GotoVelocityController


class DiffDriveVelocityController:
    """Wrapper around the velocity controller in home_robot"""

    def __init__(
        self,
        cfg,
        track_yaw=True,
    ):
        """Initilize the controller
        cfg: velocity control spec
        track_yaw: if we want to track yaw or not
        """
        self.controller = GotoVelocityController(cfg)
        self.controller.set_yaw_tracking(track_yaw)

    def set_goal(
        self,
        goal: np.ndarray,
        start: Optional[np.ndarray] = None,
        relative: bool = False,
    ):
        """Update controller goal
        goal: Desired robot base SE2 pose in global frame
        start: starting positon of the robot
        relative: if the starting position is relative or not
        """
        if relative:
            assert (
                start is not None
            ), "Start pose required if goal is relative."
            self.controller.update_pose_feedback(start)
        self.controller.update_goal(goal, relative=relative)

    def velocity_feedback_control(self, x_err, a, v_max):
        """Wrapper for using function of _velocity_feedback_control"""
        return self.controller.control._velocity_feedback_control(
            x_err, a, v_max
        )

    def forward(self, xyt, *args, **kwargs):
        """Query controller to compute velocity command
        xyt: Robot base current SE2 pose in global frame
        """
        # Update state feedback
        self.controller.update_pose_feedback(xyt)
        # Compute velocity control
        return self.controller.compute_control()
