from typing import Optional

import numpy as np
from home_robot.control.goto_controller import GotoVelocityController
from omegaconf import OmegaConf


class ContinuousController:
    """Wrapper around the velocity controller in home_robot"""

    def __init__(
        self,
        v_max=0.3,
        w_max=0.45,
        acc_lin=0.2,
        acc_ang=0.6,
        max_heading_ang=np.pi / 4,
        lin_error_tol=0.01,
        ang_error_tol=0.025,
        track_yaw=True,
    ):
        # Generate config
        cfg_dict = {
            "v_max": v_max,
            "w_max": w_max,
            "acc_lin": acc_lin,
            "acc_ang": acc_ang,
            "tol_lin": lin_error_tol,
            "tol_ang": ang_error_tol,
            "max_heading_ang": max_heading_ang,
        }
        cfg = OmegaConf.create(cfg_dict)

        # Instatiate controller
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
        """
        if relative:
            assert (
                start is not None
            ), "Start pose required if goal is relative."
            self.controller.update_pose_feedback(start)
        self.controller.update_goal(goal, relative=relative)

    def forward(self, xyt, *args, **kwargs):
        """Query controller to compute velocity command
        xyt: Robot base current SE2 pose in global frame
        """
        # Update state feedback
        self.controller.update_pose_feedback(xyt)
        # Compute velocity control
        return self.controller.compute_control()
