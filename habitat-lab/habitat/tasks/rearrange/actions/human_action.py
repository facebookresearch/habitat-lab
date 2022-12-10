from habitat.core.embodied_task import SimulatorTaskAction
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim


class HumanAction(SimulatorTaskAction):
    """
    Handles which human instance the action is applied to.
    """

    _sim: RearrangeSim

    @property
    def _human_mgr(self):
        """
        Underlying human mananger for the human instance the action is attached to.
        """

        if "agent" not in self._config or self._config.agent is None:
            return self._sim.humans_mgr[0]
        return self._sim.humans_mgr[self._config.agent]

    @property
    def _ik_helper(self):
        """
        The IK helper for this human instance.
        """

        return self._human_mgr.ik_helper

    @property
    def cur_human(self):
        """
        The human instance for this action.
        """
        return self._human_mgr.humanoid

    @property
    def cur_grasp_mgr(self):
        """
        The grasp manager for the human instance for this action.
        """
        return self._human_mgr.grasp_mgr

    @property
    def _action_arg_prefix(self) -> str:
        """
        Returns the action prefix to go in front of sensor / action names if
        there are multiple agents.
        """

        if "agent" in self._config and self._config.agent is not None:
            return f"agent_{self._config.agent}_"
        return ""
