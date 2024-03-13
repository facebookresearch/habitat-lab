# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import torch

from habitat.tasks.rearrange.multi_task.rearrange_pddl import parse_func
from habitat_baselines.common.logging import baselines_logger
from habitat_baselines.rl.hrl.hl.high_level_policy import HighLevelPolicy
from habitat_baselines.rl.ppo.policy import PolicyActionData


class FixedHighLevelPolicy(HighLevelPolicy):
    """
    Executes a fixed sequence of high-level actions as specified by the
    `solution` field of the PDDL problem file.
    :property _solution_actions: List of tuples where the first tuple element
        is the action name and the second is the action arguments. Stores a plan
        for each environment.
    """

    _solution_actions: List[List[Tuple[str, List[str]]]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._update_solution_actions(
            [self._parse_solution_actions() for _ in range(self._num_envs)]
        )

        self._next_sol_idxs = torch.zeros(self._num_envs, dtype=torch.int32)

    def _update_solution_actions(
        self, solution_actions: List[List[Tuple[str, List[str]]]]
    ) -> None:
        if len(solution_actions) == 0:
            raise ValueError(
                "Solution actions must be non-empty (if want to execute no actions, just include a no-op)"
            )
        self._solution_actions = solution_actions

    def _parse_solution_actions(self) -> List[Tuple[str, List[str]]]:
        """
        Returns the sequence of actions to execute as a list of:
        - The action name.
        - A list of the action arguments.
        """
        solution = self._pddl_prob.solution

        solution_actions = []
        for i, hl_action in enumerate(solution):
            sol_action = (
                hl_action.name,
                [x.name for x in hl_action.param_values],
            )
            solution_actions.append(sol_action)

            if self._config.add_arm_rest and i < (len(solution) - 1):
                solution_actions.append(parse_func("reset_arm(0)"))

        # Add a wait action at the end.
        solution_actions.append(parse_func("wait(30)"))

        return solution_actions

    def apply_mask(self, mask):
        """
        Apply the given mask to the next skill index.

        Args:
            mask: Binary mask of shape (num_envs, ) to be applied to the next
                skill index.
        """
        self._next_sol_idxs *= mask.cpu().view(-1)

    def _get_next_sol_idx(self, batch_idx, immediate_end):
        """
        Get the next index to be used from the list of solution actions.

        Args:
            batch_idx: The index of the current environment.

        Returns:
            The next index to be used from the list of solution actions.
        """
        if self._next_sol_idxs[batch_idx] >= len(
            self._solution_actions[batch_idx]
        ):
            baselines_logger.info(
                f"Calling for immediate end with {self._next_sol_idxs[batch_idx]}"
            )
            immediate_end[batch_idx] = True
            # Just repeat the last action.
            return len(self._solution_actions[batch_idx]) - 1
        else:
            return self._next_sol_idxs[batch_idx].item()

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        # We assign a value of 0. This is needed so that we can concatenate values in multiagent
        # policies
        return torch.zeros(rnn_hidden_states.shape[0], 1).to(
            rnn_hidden_states.device
        )

    def get_next_skill(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        plan_masks,
        deterministic,
        log_info,
    ):
        batch_size = masks.shape[0]
        next_skill = torch.zeros(batch_size)
        skill_args_data = [None for _ in range(batch_size)]
        immediate_end = torch.zeros(batch_size, dtype=torch.bool)
        for batch_idx, should_plan in enumerate(plan_masks):
            if should_plan == 1.0:
                use_idx = self._get_next_sol_idx(batch_idx, immediate_end)

                skill_name, skill_args = self._solution_actions[batch_idx][
                    use_idx
                ]
                baselines_logger.info(
                    f"Got next element of the plan with {skill_name}, {skill_args}"
                )
                if skill_name not in self._skill_name_to_idx:
                    raise ValueError(
                        f"Could not find skill named {skill_name} in {self._skill_name_to_idx}"
                    )
                next_skill[batch_idx] = self._skill_name_to_idx[skill_name]

                skill_args_data[batch_idx] = skill_args  # type: ignore[call-overload]

                self._next_sol_idxs[batch_idx] += 1

        return next_skill, skill_args_data, immediate_end, PolicyActionData()

    def filter_envs(self, curr_envs_to_keep_active):
        """
        Cleans up stateful variables of the policy so that
        they match with the active environments
        """
        self._next_sol_idxs = self._next_sol_idxs[curr_envs_to_keep_active]
        parse_solution_actions = [
            self._parse_solution_actions() for _ in range(self._num_envs)
        ]
        self._update_solution_actions(
            [
                parse_solution_actions[i]
                for i in range(curr_envs_to_keep_active.shape[0])
                if curr_envs_to_keep_active[i]
            ]
        )
