from habitat.core.embodied_task import SimulatorTaskAction
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim


class RobotAction(SimulatorTaskAction):
    """
    Handles which robot instance the action is applied to.
    """

    _sim: RearrangeSim

    @property
    def _robot_mgr(self):
        """
        Underlying robot mananger for the robot instance the action is attached to.
        """

        if self._config.agent is None:
            return self._sim.robots_mgr[0]
        return self._sim.robots_mgr[self._config.agent]

    @property
    def _ik_helper(self):
        """
        The IK helper for this robot instance.
        """

        return self._robot_mgr.ik_helper

    @property
    def cur_robot(self):
        """
        The robot instance for this action.
        """
        return self._robot_mgr.robot

    @property
    def cur_grasp_mgr(self):
        """
        The grasp manager for the robot instance for this action.
        """
        return self._robot_mgr.grasp_mgr

    @property
    def _action_arg_prefix(self) -> str:
        """
        Returns the action prefix to go in front of sensor / action names if
        there are multiple agents.
        """

        if self._config.agent is not None:
            return f"agent_{self._config.agent}_"
        return ""
