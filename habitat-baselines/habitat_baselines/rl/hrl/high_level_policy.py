from typing import Any, List, Tuple

import torch
import yaml

from habitat.tasks.rearrange.multi_task.rearrange_pddl import parse_func
from habitat_baselines.common.logging import baselines_logger


class HighLevelPolicy:
    def get_next_skill(
        self, observations, rnn_hidden_states, prev_actions, masks, plan_masks
    ) -> Tuple[torch.Tensor, List[Any], torch.BoolTensor]:
        """
        :returns: A tuple containing the next skill index, a list of arguments
            for the skill, and if the high-level policy requests immediate
            termination.
        """


class GtHighLevelPolicy:
    def __init__(self, config, task_spec_file, num_envs, skill_name_to_idx):
        with open(task_spec_file, "r") as f:
            task_spec = yaml.safe_load(f)

        self._solution_actions = []
        for i, sol_step in enumerate(task_spec["solution"]):
            sol_action = parse_func(sol_step)
            self._solution_actions.append(sol_action)
            if i < (len(task_spec["solution"]) - 1):
                self._solution_actions.append(parse_func("reset_arm(0)"))
        # Add a wait action at the end.
        self._solution_actions.append(parse_func("wait(30)"))

        self._next_sol_idxs = torch.zeros(num_envs, dtype=torch.int32)
        self._num_envs = num_envs
        self._skill_name_to_idx = skill_name_to_idx

    def apply_mask(self, mask):
        self._next_sol_idxs *= mask.cpu().view(-1)

    def get_next_skill(
        self, observations, rnn_hidden_states, prev_actions, masks, plan_masks
    ):
        next_skill = torch.zeros(self._num_envs, device=prev_actions.device)
        skill_args_data = [None for _ in range(self._num_envs)]
        immediate_end = torch.zeros(
            self._num_envs, device=prev_actions.device, dtype=torch.bool
        )
        for batch_idx, should_plan in enumerate(plan_masks):
            if should_plan == 1.0:
                if self._next_sol_idxs[batch_idx] >= len(
                    self._solution_actions
                ):
                    baselines_logger.info(
                        f"Calling for immediate end with {self._next_sol_idxs[batch_idx]}"
                    )
                    immediate_end[batch_idx] = True
                    use_idx = len(self._solution_actions) - 1
                else:
                    use_idx = self._next_sol_idxs[batch_idx].item()

                skill_name, skill_args = self._solution_actions[use_idx]
                baselines_logger.info(
                    f"Got next element of the plan with {skill_name}, {skill_args}"
                )
                if skill_name not in self._skill_name_to_idx:
                    raise ValueError(
                        f"Could not find skill named {skill_name} in {self._skill_name_to_idx}"
                    )
                next_skill[batch_idx] = self._skill_name_to_idx[skill_name]
                skill_args = skill_args.split(",")
                skill_args_data[batch_idx] = skill_args

                self._next_sol_idxs[batch_idx] += 1

        return next_skill, skill_args_data, immediate_end
