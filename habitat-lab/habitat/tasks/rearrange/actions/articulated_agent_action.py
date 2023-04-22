from habitat.core.embodied_task import SimulatorTaskAction
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim


class ArticulatedAgentAction(SimulatorTaskAction):
    """
    Handles which articulated_agent instance the action is applied to.
    """

    _sim: RearrangeSim

    def __init__(self, *args, sim, **kwargs):
        super().__init__(self, *args, sim=sim, **kwargs)
        num_agents = self._sim.num_articulated_agents
        if num_agents == 1:
            self._agent_index = 0
            self._multi_agent = False
        else:
            self._agent_index = self._config.agent_index
            self._multi_agent = True

    @property
    def _articulated_agent_mgr(self):
        """
        Underlying articulated_agent mananger for the articulated_agent instance the action is attached to.
        """
        return self._sim.agents_mgr[self._agent_index]

    @property
    def _ik_helper(self):
        """
        The IK helper for this articulated_agent instance.
        """

        return self._articulated_agent_mgr.ik_helper

    @property
    def cur_articulated_agent(self):
        """
        The articulated_agent instance for this action.
        """
        return self._articulated_agent_mgr.articulated_agent

    @property
    def cur_grasp_mgr(self):
        """
        The grasp manager for the articulated_agent instance for this action.
        """
        return self._articulated_agent_mgr.grasp_mgr

    @property
    def _action_arg_prefix(self) -> str:
        """
        Returns the action prefix to go in front of sensor / action names if
        there are multiple agents.
        """
        if not self._multi_agent:
            return ""
        return f"agent_{self._agent_index}_"
