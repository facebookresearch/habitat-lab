# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from collections import deque
from dataclasses import dataclass
from typing import List

import gym.spaces as spaces
import torch

from habitat.tasks.rearrange.multi_task.pddl_action import PddlAction
from habitat.tasks.rearrange.multi_task.pddl_logical_expr import LogicalExpr
from habitat.tasks.rearrange.multi_task.pddl_predicate import Predicate
from habitat_baselines.rl.hrl.hl.high_level_policy import HighLevelPolicy
from habitat_baselines.rl.ppo.policy import PolicyActionData


@dataclass
class PlanNode:
    """
    Represents the info in the search problem to plan a path of high-level
    actions to the goal.
    """

    cur_pred_state: List[Predicate]
    parent: "PlanNode"
    depth: int
    action: PddlAction


class PlannerHighLevelPolicy(HighLevelPolicy):
    """
    High-level policy that will plan a sequence of objects to rearrange
    actions. The `plan_idx` config parameter controls what sort of plan the
    agent will execute. The agent can either rearrange 1 of the objects, or
    both of the objects.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # This must match the predicate set in the `GlobalPredicatesSensor`.
        self._predicates_list = self._pddl_prob.get_possible_predicates()

        self._all_actions = self._setup_actions()
        self._n_actions = len(self._all_actions)
        self._max_search_depth = self._config.max_search_depth
        self._reactive_planner = self._config.is_reactive

        self._next_sol_idxs = torch.zeros(self._num_envs, dtype=torch.int32)
        self._plans: List[List[PddlAction]] = [
            [] for _ in range(self._num_envs)
        ]
        self._should_replan = torch.zeros(self._num_envs, dtype=torch.bool)
        self.gen_plan_ids_batch = torch.zeros(
            self._num_envs, dtype=torch.int32
        )
        self._plan_idx = self._config.plan_idx
        self._select_random_goal = self._config.select_random_goal
        assert self._plan_idx >= 0

        # We consider 4 possible plans, indexed from 0 to 3 which
        # correspond to moving no object (0), moving the first object (1),
        # moving the second object (2), or moving the two objects (3).
        # Let the plans be stored in an array agent_plans.
        # When select_random_goal is False, the agent executed a plan indexed by plan_idx:
        # e.g. final_plan = agent_plans[plan_idx]
        # When select_random_goal is True, the agent will select a plan at random, from
        # a subset specified by plan_idx as follows:
        # plan_index = randint(self.low_plan[plan_idx], self.high_plan[plan_idx])
        # final_final = agent_plans[plan_index].
        # As such, plan_idx specifies whether the agent should always move the
        # same object (0), one of the two objects at random (1), one or two
        # objects at random (2) or one, two or none of the objects (3).
        self.low_plan = [1, 1, 1, 0]
        self.high_plan = [1, 2, 3, 3]

    def filter_envs(self, curr_envs_to_keep_active):
        """
        Cleans up stateful variables of the policy so that
        they match with the active environments
        """
        self._should_replan = self._should_replan[curr_envs_to_keep_active]
        self.gen_plan_ids_batch = self.gen_plan_ids_batch[
            curr_envs_to_keep_active
        ]
        self._next_sol_idxs = self._next_sol_idxs[curr_envs_to_keep_active]

    def create_hl_info(self):
        return {"actions": None}

    def get_policy_action_space(
        self, env_action_space: spaces.Space
    ) -> spaces.Space:
        """
        Fetches the policy action space for learning. If we are learning the HL
        policy, it will return its custom action space for learning.
        """
        return spaces.Discrete(self._n_actions)

    def apply_mask(self, mask):
        if self._reactive_planner:
            # Replan at every step
            self._should_replan = torch.ones(mask.shape[0], dtype=torch.bool)
        else:
            # Only plan at step 0
            self._should_replan = ~mask.cpu().view(-1)

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        # We assign a value of 0. This is needed so that we can concatenate values in multiagent
        # policies
        return torch.zeros(rnn_hidden_states.shape[0], 1).to(
            rnn_hidden_states.device
        )

    def _get_solution_nodes(
        self, pred_vals, pddl_goal: LogicalExpr
    ) -> List[PlanNode]:
        """
        Based on the truth values in `pred_vals`, plan a sequence of
        PddlActions (returned in a list of `PlanNodes`) that describe how to
        get from the current state to the specified `pddl_goal`.
        """
        assert pddl_goal is not None, "Pddl goal must be set for planning."

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
        shuffled_actions = list(self._all_actions)
        random.shuffle(shuffled_actions)
        while len(stack) != 0:
            cur_node = stack.popleft()

            if cur_node.depth > self._max_search_depth:
                break

            for action in shuffled_actions:
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
                    # Unfortunately holding and not_holding are negations. The
                    # PDDL system does not currently support negations, so we
                    # have to manually handle this case.
                    if p.name == "holding":
                        pred_set = [
                            other_p
                            for other_p in pred_set
                            if not (
                                other_p.name == "not_holding"
                                and other_p._arg_values[0] == p._arg_values[1]
                            )
                        ]
                    if p.name == "not_holding":
                        pred_set = [
                            other_p
                            for other_p in pred_set
                            if not (
                                other_p.name == "holding"
                                and p._arg_values[0] == other_p._arg_values[1]
                            )
                        ]
                    if p not in pred_set:
                        pred_set.append(p)

                pred_hash = _get_pred_hash(pred_set)

                if pred_hash not in visited:
                    visited.add(pred_hash)
                    add_node = PlanNode(
                        pred_set, cur_node, cur_node.depth + 1, action
                    )
                    if pddl_goal.is_true_from_predicates(pred_set):
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

    def _get_plan(self, pred_vals, pddl_goal: LogicalExpr) -> List[PddlAction]:
        """
        :param pred_vals: Shape (num_prds,). NOT batched.
        """
        assert len(pred_vals) == len(self._predicates_list)
        sol_nodes = self._get_solution_nodes(pred_vals, pddl_goal)

        # Extract the sequence of actions that lead to the goal.
        paths = self._extract_paths(sol_nodes)

        all_ac_seqs = []
        for path in paths:
            all_ac_seqs.append([node.action for node in path])
        # Sort by the length of the action sequence
        full_plans = sorted(all_ac_seqs, key=len)
        # Each full plan will be a permutation of the other full plans.
        plans = full_plans[0]
        return plans

    def _replan(self, pred_vals, gen_plan_idx: int):
        if self._select_random_goal:
            # We select a plan at random
            index_plan = gen_plan_idx
        else:
            # Fix the plan to be the specified `plan_idx`.
            index_plan = self._plan_idx - 1

        possible_plans = [
            # Do nothing
            None,
            # Move the 1st object.
            self._pddl_prob.stage_goals["stage_1_2"],
            # Move the 2nd object.
            self._pddl_prob.stage_goals["stage_2_2"],
            # Move both objects.
            self._pddl_prob.goal,
        ]
        pddl_goal_selected = possible_plans[index_plan]
        if pddl_goal_selected is None:
            # Agent wants to do nothing, so nothing to plan.
            plans = []
        else:
            plans = self._get_plan(pred_vals, pddl_goal_selected)

        return plans

    def _get_plan_action(self, pred_vals, batch_idx: int, gen_plan_idx: int):
        if self._should_replan[batch_idx]:
            # Recompute the plan for this batch element.
            self._plans[batch_idx] = self._replan(pred_vals, gen_plan_idx)
            self._next_sol_idxs[batch_idx] = 0
            if self._plans[batch_idx] is None:
                return None

        # Progress to the next part of the plan.
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
        batch_size = masks.shape[0]
        all_pred_vals = observations["all_predicates"]
        next_skill = torch.zeros(batch_size)
        skill_args_data = [None for _ in range(batch_size)]
        immediate_end = torch.zeros(batch_size, dtype=torch.bool)

        if (~masks).sum() > 0:
            self.gen_plan_ids_batch[~masks[:, 0].cpu()] = torch.randint(
                low=self.low_plan[self._plan_idx - 1],
                high=self.high_plan[self._plan_idx - 1] + 1,
                size=[(~masks).int().sum().item()],
                dtype=torch.int32,
            )

        for batch_idx, should_plan in enumerate(plan_masks):
            if should_plan != 1.0:
                continue
            gen_plan_idx = self.gen_plan_ids_batch[batch_idx]
            cur_ac = self._get_plan_action(
                all_pred_vals[batch_idx], batch_idx, gen_plan_idx
            )
            if cur_ac is not None:
                next_skill[batch_idx] = self._skill_name_to_idx[cur_ac.name]
                skill_args_data[batch_idx] = [param.name for param in cur_ac.param_values]  # type: ignore[call-overload]
            else:
                # If we have no next action, do nothing.
                next_skill[batch_idx] = self._skill_name_to_idx["wait"]
                # Wait 1 step.
                skill_args_data[batch_idx] = ["1"]  # type: ignore[call-overload]
        return (
            next_skill,
            skill_args_data,
            immediate_end,
            PolicyActionData(),
        )
