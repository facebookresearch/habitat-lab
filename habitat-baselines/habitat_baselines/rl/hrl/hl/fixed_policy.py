# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import torch
import yaml

from habitat.config.default import get_full_habitat_config_path
from habitat.tasks.rearrange.multi_task.rearrange_pddl import parse_func
from habitat_baselines.common.logging import baselines_logger
from habitat_baselines.rl.hrl.hl.high_level_policy import HighLevelPolicy


class FixedHighLevelPolicy(HighLevelPolicy):
    """
    Executes a fixed sequence of high-level actions as specified by the
    `solution` field of the PDDL problem file.
    :property _solution_actions: List of tuples were first tuple element is the
        action name and the second is the action arguments.
    """

    _solution_actions: List[Tuple[str, List[str]]]

    def __init__(self, config, task_spec_file, num_envs, skill_name_to_idx):
        """
        Initialize the `FixedHighLevelPolicy` object.

        Args:
            config: Config object containing the configurations for the agent.
            task_spec_file: Path to the task specification file.
            num_envs: Number of parallel environments.
            skill_name_to_idx: Dictionary mapping skill names to skill indices.
        """
        with open(get_full_habitat_config_path(task_spec_file), "r") as f:
            task_spec = yaml.safe_load(f)

        self._num_envs = num_envs
        self._skill_name_to_idx = skill_name_to_idx
        self._solution_actions = self._parse_solution_actions(
            config, task_spec, task_spec_file
        )

        self._next_sol_idxs = torch.zeros(num_envs, dtype=torch.int32)

    def _parse_solution_actions(self, config, task_spec, task_spec_file):
        if "solution" not in task_spec:
            raise ValueError(
                f"The ground truth task planner only works when the task solution is hard-coded in the PDDL problem file at {task_spec_file}."
            )

        solution_actions = []
        for i, sol_step in enumerate(task_spec["solution"]):
            sol_action = parse_func(sol_step)
            solution_actions.append(sol_action)

            if config.add_arm_rest and i < (len(task_spec["solution"]) - 1):
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
        if self._next_sol_idxs[batch_idx] >= len(self._solution_actions):
            baselines_logger.info(
                f"Calling for immediate end with {self._next_sol_idxs[batch_idx]}"
            )
            immediate_end[batch_idx] = True
            return len(self._solution_actions) - 1
        else:
            return self._next_sol_idxs[batch_idx].item()

    def get_next_skill(
        self, observations, rnn_hidden_states, prev_actions, masks, plan_masks
    ):
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
        """
        next_skill = torch.zeros(self._num_envs)
        skill_args_data = [None for _ in range(self._num_envs)]
        immediate_end = torch.zeros(self._num_envs, dtype=torch.bool)
        for batch_idx, should_plan in enumerate(plan_masks):
            if should_plan == 1.0:
                use_idx = self._get_next_sol_idx(batch_idx, immediate_end)

                skill_name, skill_args = self._solution_actions[use_idx]
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

        return next_skill, skill_args_data, immediate_end
