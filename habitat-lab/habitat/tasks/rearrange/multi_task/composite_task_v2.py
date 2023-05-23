#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Union, cast

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.datasets.rearrange.rearrange_dataset import RearrangeDatasetV0
from habitat.tasks.rearrange.multi_task.pddl_domain import PddlDomain
from habitat.tasks.rearrange.multi_task.pddl_logical_expr import (
    LogicalExpr,
    LogicalExprType,
)
from habitat.tasks.rearrange.multi_task.pddl_predicate import Predicate
from habitat.tasks.rearrange.multi_task.rearrange_pddl import (
    PddlEntity,
    SimulatorObjectType,
)
from habitat.tasks.rearrange.rearrange_task import RearrangeTask


@registry.register_task(name="RearrangeCompositeTask-v2")
class CompositeTaskV2(RearrangeTask):
    """
    Automatically sets up the PDDL problem.
    """

    def __init__(self, *args, config, dataset=None, **kwargs):
        self.pddl_problem = PddlDomain(
            config.pddl_domain_def,
            config,
        )

        super().__init__(config=config, *args, dataset=dataset, **kwargs)

    @property
    def goal(self) -> LogicalExpr:
        return self._goal

    @property
    def stage_goals(self) -> Dict[str, LogicalExpr]:
        return self._subgoal_preds

    def reset(self, episode: Episode):
        super().reset(episode, fetch_observations=False)
        self.pddl_problem.bind_to_instance(
            self._sim, cast(RearrangeDatasetV0, self._dataset), self, episode
        )
        targ_type = self.pddl_problem.expr_types[
            SimulatorObjectType.GOAL_ENTITY.value
        ]
        obj_type = self.pddl_problem.expr_types[
            SimulatorObjectType.MOVABLE_ENTITY.value
        ]
        robot_type = self.pddl_problem.expr_types[
            SimulatorObjectType.ROBOT_ENTITY.value
        ]

        id_to_name = {}
        for k, i in self.pddl_problem.sim_info.obj_ids.items():
            id_to_name[i] = k

        target_to_sim_id = self._sim.get_targets()[0]

        obj_goal_assocs = {}
        for (
            target_name,
            target_idx,
        ) in self.pddl_problem.sim_info.target_ids.items():
            goal_entity = PddlEntity(target_name, targ_type)
            self.pddl_problem.register_episode_entity(goal_entity)

            # Find the object that should be placed at this goal.
            sim_obj_id = target_to_sim_id[target_idx]
            obj_handle = id_to_name[sim_obj_id]
            obj_entity = PddlEntity(obj_handle, obj_type)
            self.pddl_problem.register_episode_entity(obj_entity)
            obj_goal_assocs[goal_entity] = obj_entity

        goal_preds = [
            self.pddl_problem.parse_predicate(
                f"at({obj.name},{goal.name})", self.pddl_problem.all_entities
            )
            for goal, obj in obj_goal_assocs.items()
        ]

        not_holding_preds = []
        for agent_i in range(self._sim.num_articulated_agents):
            robot_name = f"robot_{agent_i}"

            self.pddl_problem.register_episode_entity(
                PddlEntity(robot_name, robot_type)
            )
            not_holding_preds.append(
                self.pddl_problem.parse_predicate(
                    f"not_holding({robot_name})",
                    self.pddl_problem.all_entities,
                )
            )

        self._subgoal_preds = {}
        for i in range(self.max_num_objects):
            preds: List[Union[LogicalExpr, Predicate]] = []
            if i < len(goal_preds):
                goal_pred = goal_preds[i]
                preds = [goal_pred, *not_holding_preds]
            self._subgoal_preds[f"move_obj{i}"] = LogicalExpr(
                LogicalExprType.AND, preds, [], None
            )

        self._goal = LogicalExpr(
            LogicalExprType.AND, [*goal_preds, *not_holding_preds], [], None
        )

        self._sim.maybe_update_articulated_agent()
        return self._get_observations(episode)
