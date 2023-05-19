# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import torch

from habitat.tasks.rearrange.multi_task.rearrange_pddl import parse_func
from habitat_baselines.common.logging import baselines_logger
from habitat_baselines.rl.hrl.hl.high_level_policy import HighLevelPolicy


class SocNavHumanHighLevelPolicy(HighLevelPolicy):
    """
    Executes a fixed sequence of high-level actions as specified by the
    `solution` field of the PDDL problem file.
    :property _solution_actions: List of tuples were first tuple element is the
        action name and the second is the action arguments.
    """

    _solution_actions: List[Tuple[str, List[str]]]

    def __init__(self, *args, **kwargs):
        print("RUNNING SocNavHumanHighLevelPolicy!")
        super().__init__(*args, **kwargs)

        # self._solution_actions = self._parse_solution_actions(
        #     self._pddl_prob.solution
        # )
        self._solution_actions = [['nonexistent_goal', 'robot_0']]

        self._next_sol_idxs = torch.zeros(self._num_envs, dtype=torch.int32)
        #self.skill_called_count = 0

    def _parse_solution_actions(self, solution):
        solution_actions = []
        for i, hl_action in enumerate(solution):
            sol_action = (
                hl_action.name,
                [x.name for x in hl_action.param_values],
            )
            # Filter out the correct action for the robot.
            robot_id = "robot_" + self._agent_name.split("_")[1]
            #if robot_id in sol_action[1]:
            if True:
                solution_actions.append(sol_action)

            if self._config.add_arm_rest and i < (len(solution) - 1):
                solution_actions.append(parse_func("reset_arm(0)"))

        # Add a wait action at the end.
        solution_actions.append(parse_func("wait(30)"))
        #import ipdb; ipdb.set_trace()
        return solution_actions

    # def get_termination(self,
    #     observations,
    #     rnn_hidden_states,
    #     prev_actions,
    #     masks,
    #     cur_skills,
    #     log_info,):
    #     if self.skill_called_count >=2 :
    #         print("get get_termination called!")
    #         return torch.ones(self._num_envs, dtype=torch.bool)
    #     else:
    #         return torch.zeros(self._num_envs, dtype=torch.bool)

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
        next_skill = torch.zeros(self._num_envs)
        skill_args_data = [None for _ in range(self._num_envs)]
        immediate_end = torch.zeros(self._num_envs, dtype=torch.bool)
        for batch_idx, should_plan in enumerate(plan_masks):
            # print("should plan?", should_plan)
            if should_plan == 1.0:
                next_skill[batch_idx] = self._skill_name_to_idx["nav_to_goal"]
                skill_args_data[batch_idx] = ['nonexistent_goal', 'robot_1'] #['goal0|0', 'robot_0']#['nonexistent_goal', 'robot_0']#['goal0|0', 'robot_0']#['nonexistent_goal', 'robot_0'] #[['goal0|0', 'robot_0']]  # type: ignore[call-overload]

                self._next_sol_idxs[batch_idx] += 1
                # if self.skill_called_count >0:
                #     print("immeidate end called!")
                #     immediate_end[batch_idx] = 1

            # if should_plan == 1.0:
            #     use_idx = self._get_next_sol_idx(batch_idx, immediate_end)

            #     skill_name, skill_args = self._solution_actions[use_idx]
            #     baselines_logger.info(
            #         f"Got next element of the plan with {skill_name}, {skill_args}"
            #     )
            #     if skill_name not in self._skill_name_to_idx:
            #         raise ValueError(
            #             f"Could not find skill named {skill_name} in {self._skill_name_to_idx}"
            #         )
            #     next_skill[batch_idx] = self._skill_name_to_idx[skill_name]

            #     skill_args_data[batch_idx] = skill_args  # type: ignore[call-overload]

            #     self._next_sol_idxs[batch_idx] += 1
            #     import ipdb; ipdb.set_trace()
        #print("self._next_sol_idxs", self._next_sol_idxs)
        #self.skill_called_count +=1
        #breakpoint()
        return next_skill, skill_args_data, immediate_end, {}