import os.path as osp

import torch
import yaml

from habitat.tasks.rearrange.multi_task.rearrange_pddl import parse_func


class HighLevelPolicy:
    def get_next_skill(
        self, observations, rnn_hidden_states, prev_actions, masks, plan_masks
    ):
        pass


class GtHighLevelPolicy:
    def __init__(self, config, task_spec_file, num_envs, skill_name_to_idx):
        with open(task_spec_file, "r") as f:
            task_spec = yaml.safe_load(f)
        self._solution_actions = [
            parse_func(sol_step) for sol_step in task_spec["solution"]
        ]
        self._next_sol_idxs = torch.zeros(num_envs, dtype=torch.int32)
        self._num_envs = num_envs
        self._skill_name_to_idx = skill_name_to_idx

    def apply_mask(self, mask):
        self._next_sol_idxs *= mask.cpu().view(-1)

    def get_next_skill(
        self, observations, rnn_hidden_states, prev_actions, masks, plan_masks
    ):
        next_skill = torch.zeros(self._num_envs, device=prev_actions.device)
        skill_args_tensor = torch.zeros(self._num_envs, dtype=torch.int32)
        for batch_idx, should_plan in enumerate(plan_masks):
            if should_plan == 1.0:
                skill_name, skill_args = self._solution_actions[
                    self._next_sol_idxs[batch_idx].item()
                ]
                next_skill[batch_idx] = self._skill_name_to_idx[skill_name]
                # Need to convert the name into the corresponding target index.
                # For now use a hack and assume the name correctly indexes
                if len(skill_args) > 0:
                    targ_idx = int(skill_args.split("|")[1])
                    skill_args_tensor[batch_idx] = targ_idx
                self._next_sol_idxs[batch_idx] += 1
        return next_skill, skill_args_tensor
