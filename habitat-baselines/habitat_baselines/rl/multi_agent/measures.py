from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.tasks.rearrange.multi_task.composite_sensors import (
    CompositeSubgoalReward,
)
from habitat.tasks.rearrange.multi_task.pddl_logical_expr import LogicalExpr
from habitat.tasks.rearrange.multi_task.pddl_predicate import Predicate
from habitat.tasks.rearrange.utils import coll_name_matches


@registry.register_measure
class DidAgentsCollide(Measure):
    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "did_collide"

    def reset_metric(self, *args, **kwargs):
        self.update_metric(
            *args,
            **kwargs,
        )

    def update_metric(self, *args, task, **kwargs):
        sim = task._sim
        sim.perform_discrete_collision_detection()
        contact_points = sim.get_physics_contact_points()
        found_contact = False

        agent_ids = [
            articulated_agent.sim_obj.object_id
            for articulated_agent in sim.agents_mgr.articulated_agents_iter
        ]
        if len(agent_ids) != 2:
            raise ValueError("Sensor only supports 2 agents")

        for cp in contact_points:
            if coll_name_matches(cp, agent_ids[0]) and coll_name_matches(
                cp, agent_ids[1]
            ):
                found_contact = True

        self._metric = found_contact


@registry.register_measure
class NumCollisions(Measure):
    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "num_collisions"

    def reset_metric(self, *args, task, **kwargs):
        task.measurements.check_measure_dependencies(
            self._get_uuid(), [DidAgentsCollide._get_uuid()]
        )
        self._metric = 0
        self.update_metric(
            *args,
            task=task,
            **kwargs,
        )

    def update_metric(self, *args, task, **kwargs):
        did_collide = task.measurements.measures[
            DidAgentsCollide._get_uuid()
        ].get_metric()
        self._metric += int(did_collide)


@registry.register_measure
class CooperateSubgoalReward(CompositeSubgoalReward):
    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "cooperate_subgoal_reward"

    def __init__(self, *args, config, **kwargs):
        super().__init__(*args, config=config, **kwargs)
        self._end_on_collide = config.end_on_collide
        self._collide_penalty = config.collide_penalty

    def reset_metric(self, *args, task, **kwargs):
        task.measurements.check_measure_dependencies(
            self._get_uuid(), [DidAgentsCollide._get_uuid()]
        )
        super().reset_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        super().update_metric(*args, task=task, **kwargs)
        did_collide = task.measurements.measures[
            DidAgentsCollide._get_uuid()
        ].get_metric()

        if did_collide and self._end_on_collide:
            task.should_end = True
            self._metric -= self._collide_penalty


@registry.register_measure
class AgentBlameMeasure(Measure):
    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "agent_blame"

    def _get_goal_values(self, task):
        return {
            goal_name: task.pddl_problem.is_expr_true(expr)
            for goal_name, expr in task.pddl_problem.stage_goals.items()
        }

    def reset_metric(self, *args, task, **kwargs):
        self._prev_goal_states = self._get_goal_values(task)
        self._agent_blames = {}
        for agent in range(2):
            for k in self._prev_goal_states:
                self._agent_blames[f"{agent}_{k}"] = False
        self.update_metric(
            *args,
            task=task,
            **kwargs,
        )

    def update_metric(self, *args, task, **kwargs):
        cur_goal_states = self._get_goal_values(task)

        changed_goals = []
        for goal_name, is_sat in cur_goal_states.items():
            if is_sat and not self._prev_goal_states[goal_name]:
                changed_goals.append(goal_name)

        for k in changed_goals:
            any_true = False
            sub_goal = task.pddl_problem.stage_goals[k]
            for agent in range(2):
                pddl_action = task.actions[f"agent_{agent}_pddl_apply_action"]
                if pddl_action._prev_action is None:
                    continue
                if pddl_action.was_prev_action_invalid:
                    continue
                post_cond_in_sub_goal = (
                    len(
                        AgentBlameMeasure._logical_expr_contains(
                            sub_goal, pddl_action._prev_action._post_cond
                        )
                    )
                    > 0
                )
                self._agent_blames[f"{agent}_{k}"] = post_cond_in_sub_goal
                any_true = any_true or post_cond_in_sub_goal

            # If neither of the agents can be attributed, then attribute both agents.
            if not any_true:
                for agent in range(2):
                    pddl_action = task.actions[
                        f"agent_{agent}_pddl_apply_action"
                    ]
                    if pddl_action._prev_action is None:
                        continue
                    if pddl_action.was_prev_action_invalid:
                        continue
                    self._agent_blames[f"{agent}_{k}"] = True

        self._prev_goal_states = cur_goal_states
        self._metric = self._agent_blames

    @staticmethod
    def _logical_expr_contains(expr, preds):
        ret = []
        for sub_expr in expr.sub_exprs:
            if isinstance(sub_expr, LogicalExpr):
                ret.extend(
                    AgentBlameMeasure._logical_expr_contains(sub_expr, preds)
                )
            elif isinstance(sub_expr, Predicate):
                for pred in preds:
                    if sub_expr.compact_str == pred.compact_str:
                        ret.append(sub_expr)
                        break
            else:
                raise ValueError()
        return ret
