from typing import Any, Dict, List, Tuple

import gym.spaces as spaces
import torch
import torch.nn as nn

from habitat.tasks.rearrange.multi_task.pddl_domain import PddlProblem


class HighLevelPolicy(nn.Module):
    """
    High level policy that selects from low-level skills.
    """

    def __init__(
        self,
        config,
        pddl_problem: PddlProblem,
        num_envs: int,
        skill_name_to_idx: Dict[int, str],
        observation_space: spaces.Space,
        action_space: spaces.Space,
    ):
        super().__init__()
        self._config = config
        self._pddl_prob = pddl_problem
        self._num_envs = num_envs
        self._skill_name_to_idx = skill_name_to_idx
        self._obs_space = observation_space
        self._device = None

    def to(self, device):
        self._device = device
        return super().to(device)

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        raise NotImplementedError()

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
    def num_recurrent_layers(self):
        return 0

    def parameters(self):
        return iter([nn.Parameter(torch.zeros((1,), device=self._device))])

    def get_policy_action_space(
        self, env_action_space: spaces.Space
    ) -> spaces.Space:
        return env_action_space

    def get_next_skill(
        self,
        observations,
        rnn_hidden_states: torch.Tensor,
        prev_actions: torch.Tensor,
        masks: torch.Tensor,
        plan_masks: torch.Tensor,
        deterministic: bool,
        log_info: List[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, List[Any], torch.BoolTensor, Dict[str, Any]]:
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
            - Information for PolicyActionData
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
