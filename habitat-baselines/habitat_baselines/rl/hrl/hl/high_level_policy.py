from typing import Any, Dict, List, Optional, Tuple

import gym.spaces as spaces
import torch
import torch.nn as nn

from habitat.tasks.rearrange.multi_task.pddl_action import PddlAction
from habitat.tasks.rearrange.multi_task.pddl_domain import PddlProblem
from habitat_baselines.rl.ppo.policy import PolicyActionData


class HighLevelPolicy(nn.Module):
    """
    High level policy that selects from low-level skills.
    """

    def __init__(
        self,
        config,
        pddl_problem: PddlProblem,
        num_envs: int,
        skill_name_to_idx: Dict[str, int],
        observation_space: spaces.Space,
        action_space: spaces.Space,
        aux_loss_config=None,
        agent_name: Optional[str] = None,
    ):
        super().__init__()
        self._config = config
        self._pddl_prob = pddl_problem
        self._num_envs = num_envs
        self._skill_name_to_idx = skill_name_to_idx
        self._obs_space = observation_space
        self._device = None
        self._agent_name = agent_name
        self._action_space = action_space

    def to(self, device):
        self._device = device
        return super().to(device)

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        raise NotImplementedError()

    @property
    def should_load_agent_state(self) -> bool:
        """
        If we need to load the state dict of the high-level policy.
        """
        return False

    def on_envs_pause(self, envs_to_pause: List[int]) -> None:
        pass

    def evaluate_actions(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        action,
        rnn_build_seq_info,
    ):
        raise NotImplementedError()

    @property
    def hidden_state_shape(self):
        return (
            self.num_recurrent_layers,
            self.recurrent_hidden_size,
        )

    @property
    def hidden_state_shape_lens(self):
        return [self.recurrent_hidden_size]

    @property
    def policy_action_space_shape_lens(self):
        return [self._action_space]

    @property
    def policy_action_space(self):
        return self._action_space

    @property
    def num_recurrent_layers(self):
        return 0

    @property
    def recurrent_hidden_size(self):
        return 0

    def parameters(self):
        return iter([nn.Parameter(torch.zeros((1,), device=self._device))])

    def get_next_skill(
        self,
        observations,
        rnn_hidden_states: torch.Tensor,
        prev_actions: torch.Tensor,
        masks: torch.Tensor,
        plan_masks: torch.Tensor,
        deterministic: bool,
        log_info: List[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, List[Any], torch.BoolTensor, PolicyActionData]:
        """
        Get the next skill to be executed.

        Args:
            observations: Current observations.
            rnn_hidden_states: Current hidden states of the RNN.
            prev_actions: Previous actions taken.
            masks: Binary masks indicating which environment(s) are active.
            plan_masks: Binary masks indicating which environment(s) should
                plan the next skill.

        Returns:
            A tuple containing:
            - next_skill: Next skill to be executed.
            - skill_args_data: Arguments for the next skill.
            - immediate_end: Binary masks indicating which environment(s) should
                end immediately.
            - PolicyActionData information for learning.
        """
        raise NotImplementedError()

    def apply_mask(self, mask: torch.Tensor) -> None:
        """
        Called before every step with the mask information at the current step.
        """

    def get_policy_components(self) -> List[nn.Module]:
        """
        Gets the torch modules that are in the HL policy architecture.
        """

        return []

    def get_termination(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_skills,
        log_info,
    ) -> torch.BoolTensor:
        """
        Can force the currently executing skill to terminate.
        In the base HighLevelPolicy, the skill always continues.

        Returns: A binary tensor where 1 indicates the current skill should
            terminate and 0 indicates the skill can continue.
        """

        return torch.zeros(self._num_envs, dtype=torch.bool)

    def _setup_actions(self) -> List[PddlAction]:
        """
        Returns the list of all actions this agent can execute.
        """

        # In the PDDL domain, the agents are referred to as robots.
        robot_id = "robot_" + self._agent_name.split("_")[1]

        if robot_id in self._pddl_prob.all_entities:
            # There are potentially multiple robots and we need to filter by
            # just this robot.
            filter_set = [self._pddl_prob.get_entity(robot_id)]
        else:
            # There are not other robot entities, so no need to filter.
            filter_set = []
        all_actions = self._pddl_prob.get_possible_actions(
            filter_entities=filter_set,
            allowed_action_names=self._config.allowed_actions,
        )
        if not self._config.allow_other_place:
            all_actions = [
                ac
                for ac in all_actions
                if (
                    ac.name != "place"
                    or ac.param_values[0].name in ac.param_values[1].name
                )
            ]
        return all_actions
