from habitat.core.embodied_task import SimulatorTaskAction
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim


class HumanAction(SimulatorTaskAction):
    """
    Handles which human instance the action is applied to.
    """

    _sim: RearrangeSim
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        if "agent_index" not in self._config or self._config.agent_index is None:
            self._agent_index = 0
            self._multi_agent = False
        else:
            self._agent_index = self._config.agent_index
            self._multi_agent = True

    @property
    def _human_mgr(self):
        """
        Underlying human mananger for the human instance the action is attached to.
        """
        return self._sim.agents_mgr[self._agent_index]

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
        return self._human_mgr.agent

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

        if not self._multi_agent:
            return ""
        return f"agent_{self._agent_index}_"
