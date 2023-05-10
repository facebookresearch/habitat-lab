# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import deque
from dataclasses import dataclass
from typing import List

import torch

from habitat.tasks.rearrange.multi_task.pddl_action import PddlAction
from habitat.tasks.rearrange.multi_task.pddl_predicate import Predicate
from habitat_baselines.rl.hrl.hl.high_level_policy import HighLevelPolicy


@dataclass
class PlanNode:
    cur_pred_state: List[Predicate]
    parent: "PlanNode"
    depth: int
    action: PddlAction


class PlannerHighLevelPolicy(HighLevelPolicy):
    """
    Executes a fixed sequence of high-level actions as specified by the
    `solution` field of the PDDL problem file.
    :property _solution_actions: List of tuples were first tuple element is the
        action name and the second is the action arguments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # This must match the predicate set in the `GlobalPredicatesSensor`.
        self._predicates_list = self._pddl_prob.get_possible_predicates()
        self._all_actions = self._setup_actions()
        self._max_search_depth = self._config.max_search_depth
        self._reactive_planner = self._config.is_reactive

        self._next_sol_idxs = torch.zeros(self._num_envs, dtype=torch.int32)
        self._plans: List[List[PddlAction]] = [
            [] for _ in range(self._num_envs)
        ]
        self._should_replan = torch.zeros(self._num_envs, dtype=torch.bool)

    def apply_mask(self, mask):
        if self._reactive_planner:
            # Replan at every step
            self._should_replan = torch.ones(self._num_envs, dtype=torch.bool)
        else:
            # Only plan at step 0
            self._should_replan = ~mask.cpu().view(-1)

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        # We assign a value of 0. This is needed so that we can concatenate values in multiagent
        # policies
        return torch.zeros(rnn_hidden_states.shape[0], 1).to(
            rnn_hidden_states.device
        )

    def _get_solution_nodes(self, pred_vals):
        # The true predicates at the current state
        start_true_preds = [
            pred
            for is_valid, pred in zip(pred_vals, self._predicates_list)
            if (is_valid == 1.0)
        ]

        def _is_pred_at(pred, robot_type):
            return (
                pred.name == "robot_at" and pred._arg_values[-1] == robot_type
            )

        def _get_pred_hash(preds):
            return ",".join(sorted([p.compact_str for p in preds]))

        stack = deque([PlanNode(start_true_preds, None, 0, None)])
        visited = {_get_pred_hash(start_true_preds)}
        sol_nodes = []
        while len(stack) != 0:
            cur_node = stack.popleft()

            if cur_node.depth > self._max_search_depth:
                break

            for action in self._all_actions:
                if not action.is_precond_satisfied_from_predicates(
                    cur_node.cur_pred_state
                ):
                    continue

                # Use set so we filter out duplicate predicates.
                pred_set = list(cur_node.cur_pred_state)
                if "nav" in action.name:
                    # Remove the at precondition, since we are walking somewhere else
                    robot_to_nav = action._param_values[-1]
                    pred_set = [
                        pred
                        for pred in pred_set
                        if not _is_pred_at(pred, robot_to_nav)
                    ]
                for p in action.post_cond:
                    if p not in pred_set:
                        pred_set.append(p)

                pred_hash = _get_pred_hash(pred_set)

                if pred_hash not in visited:
                    visited.add(pred_hash)
                    add_node = PlanNode(
                        pred_set, cur_node, cur_node.depth + 1, action
                    )
                    if self._pddl_prob.goal.is_true_from_predicates(pred_set):
                        # Found a goal, we can stop searching.
                        sol_nodes.append(add_node)
                    else:
                        stack.append(add_node)
        return sol_nodes

    def _extract_paths(self, sol_nodes):
        paths = []
        for sol_node in sol_nodes:
            cur_node = sol_node
            path = []
            while cur_node.parent is not None:
                path.append(cur_node)
                cur_node = cur_node.parent
            paths.append(path[::-1])
        return paths

    def _get_all_plans(self, pred_vals):
        """
        :param pred_vals: Shape (num_prds,). NOT batched.
        """
        assert len(pred_vals) == len(self._predicates_list)
        sol_nodes = self._get_solution_nodes(pred_vals)

        # Extract the paths to the goals.
        paths = self._extract_paths(sol_nodes)

        all_ac_seqs = []
        for path in paths:
            all_ac_seqs.append([node.action for node in path])
        # Sort by the length of the action sequence
        full_plans = sorted(all_ac_seqs, key=len)

        # Each full plan will be a permutation of the other full plans.
        plans = full_plans[1:]
        # Only extract subsequences from 1 of the plans.
        full_plan = full_plans[0]
        for num_subplans in range(
            0, len(full_plan) + 1, self._config.plan_split_len
        ):
            if num_subplans == 0:
                plans.append([])
                continue
            for start_i in range(0, len(full_plan), num_subplans):
                plans.append(full_plan[start_i : start_i + num_subplans])
        return plans

    def _replan(self, pred_vals):
        plans = self._get_all_plans(pred_vals)

        # Just return the shortest plan for now.
        return plans[self._config.plan_idx]

    def _get_plan_action(self, pred_vals, batch_idx):
        if self._should_replan[batch_idx]:
            self._plans[batch_idx] = self._replan(pred_vals)
            self._next_sol_idxs[batch_idx] = 0
        cur_plan = self._plans[batch_idx]

        cur_idx = self._next_sol_idxs[batch_idx]
        if cur_idx >= len(cur_plan):
            cur_ac = None
        else:
            cur_ac = cur_plan[cur_idx]

        self._next_sol_idxs[batch_idx] += 1
        return cur_ac

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
        all_pred_vals = observations["all_predicates"]
        next_skill = torch.zeros(self._num_envs)
        skill_args_data = [None for _ in range(self._num_envs)]
        immediate_end = torch.zeros(self._num_envs, dtype=torch.bool)
        for batch_idx, should_plan in enumerate(plan_masks):
            if should_plan != 1.0:
                continue
            cur_ac = self._get_plan_action(all_pred_vals[batch_idx], batch_idx)
            if cur_ac is not None:
                next_skill[batch_idx] = self._skill_name_to_idx[cur_ac.name]
                skill_args_data[batch_idx] = [param.name for param in cur_ac.param_values]  # type: ignore[call-overload]
            else:
                # If we have no next action, do nothing.
                next_skill[batch_idx] = self._skill_name_to_idx["wait"]
                # Wait 1 step.
                skill_args_data[batch_idx] = ["1"]  # type: ignore[call-overload]
        return next_skill, skill_args_data, immediate_end, {}
