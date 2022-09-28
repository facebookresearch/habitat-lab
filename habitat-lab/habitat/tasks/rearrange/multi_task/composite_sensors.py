#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import List

import numpy as np
from gym import spaces

from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.rearrange.rearrange_sensors import (
    EndEffectorToObjectDistance,
    ObjectToGoalDistance,
    RearrangeReward,
)


@registry.register_sensor
class GlobalPredicatesSensor(Sensor):
    def __init__(self, sim, config, *args, task, **kwargs):
        self._task = task
        self._sim = sim
        self._predicates_list = None
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return "all_predicates"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    @property
    def predicates_list(self):
        if self._predicates_list is None:
            self._predicates_list = (
                self._task.pddl_problem.get_possible_predicates()
            )
        return self._predicates_list

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(len(self.predicates_list),),
            low=0,
            high=1,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        sim_info = self._task.pddl_problem.sim_info
        truth_values = [p.is_true(sim_info) for p in self.predicates_list]
        return np.array(truth_values, dtype=np.float32)


@registry.register_measure
class MoveObjectsReward(RearrangeReward):
    """
    A reward based on L2 distances to object/goal.
    """

    cls_uuid: str = "move_obj_reward"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return MoveObjectsReward.cls_uuid

    def __init__(self, *args, **kwargs):
        self._cur_rearrange_step = 0
        super().__init__(*args, **kwargs)

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self._cur_rearrange_step = 0
        self._prev_holding_obj = False
        self._did_give_pick_reward = {}
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                ObjectToGoalDistance.cls_uuid,
                EndEffectorToObjectDistance.cls_uuid,
            ],
        )

        to_goal = task.measurements.measures[
            ObjectToGoalDistance.cls_uuid
        ].get_metric()
        to_obj = task.measurements.measures[
            EndEffectorToObjectDistance.cls_uuid
        ].get_metric()
        self._prev_measures = (to_obj, to_goal)

        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        super().update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )
        idxs, _ = self._sim.get_targets()
        targ_obj_idx = idxs[self._cur_rearrange_step]
        abs_targ_obj_idx = self._sim.scene_obj_ids[targ_obj_idx]
        targ_obj_idx = str(targ_obj_idx)
        num_targs = len(idxs)

        to_goal = task.measurements.measures[
            ObjectToGoalDistance.cls_uuid
        ].get_metric()
        to_obj = task.measurements.measures[
            EndEffectorToObjectDistance.cls_uuid
        ].get_metric()

        is_holding_obj = self._sim.grasp_mgr.snap_idx == abs_targ_obj_idx
        if is_holding_obj:
            dist = to_goal[targ_obj_idx]
            dist_diff = (
                self._prev_measures[1][targ_obj_idx] - to_goal[targ_obj_idx]
            )
        else:
            dist = to_obj[targ_obj_idx]
            dist_diff = (
                self._prev_measures[0][targ_obj_idx] - to_obj[targ_obj_idx]
            )

        if (
            is_holding_obj
            and not self._prev_holding_obj
            and self._cur_rearrange_step not in self._did_give_pick_reward
        ):
            self._metric += self._config.pick_reward
            self._did_give_pick_reward[self._cur_rearrange_step] = True

        if (
            dist < self._config.success_dist
            and not is_holding_obj
            and self._cur_rearrange_step < num_targs
        ):
            self._metric += self._config.single_rearrange_reward
            self._cur_rearrange_step += 1
            self._cur_rearrange_step = min(
                self._cur_rearrange_step, num_targs - 1
            )

        self._metric += self._config.dist_reward * dist_diff
        self._prev_measures = (to_obj, to_goal)
        self._prev_holding_obj = is_holding_obj


@registry.register_measure
class DoesWantTerminate(Measure):
    cls_uuid: str = "does_want_terminate"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return DoesWantTerminate.cls_uuid

    def reset_metric(self, *args, **kwargs):
        self.update_metric(*args, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        self._metric = task.actions["rearrange_stop"].does_want_terminate


@registry.register_measure
class CompositeBadCalledTerminate(Measure):
    cls_uuid: str = "composite_bad_called_terminate"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return CompositeBadCalledTerminate.cls_uuid

    def reset_metric(self, *args, task, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [DoesWantTerminate.cls_uuid, CompositeSuccess.cls_uuid],
        )
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        does_action_want_stop = task.measurements.measures[
            DoesWantTerminate.cls_uuid
        ].get_metric()
        is_succ = task.measurements.measures[
            CompositeSuccess.cls_uuid
        ].get_metric()

        self._metric = (not is_succ) and does_action_want_stop


@registry.register_measure
class CompositeSuccess(Measure):
    """
    Did satisfy all the goal predicates?
    """

    cls_uuid: str = "composite_success"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(**kwargs)
        self._sim = sim
        self._config = config

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return CompositeSuccess.cls_uuid

    def reset_metric(self, *args, task, **kwargs):
        if self._config.must_call_stop:
            task.measurements.check_measure_dependencies(
                self.uuid,
                [DoesWantTerminate.cls_uuid],
            )
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, episode, task, observations, **kwargs):
        self._metric = task.pddl_problem.is_expr_true(task.pddl_problem.goal)

        if self._config.must_call_stop:
            does_action_want_stop = task.measurements.measures[
                DoesWantTerminate.cls_uuid
            ].get_metric()
            self._metric = self._metric and does_action_want_stop
        else:
            does_action_want_stop = False

        if does_action_want_stop:
            task.should_end = True


@registry.register_measure
class CompositeStageGoals(Measure):
    """
    Adds to the metrics `[task_NAME]_success`: Did the agent complete a
        particular stage defined in `stage_goals`.
    """

    _stage_succ: List[str]
    cls_uuid: str = "composite_stage_goals"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return CompositeStageGoals.cls_uuid

    def reset_metric(self, *args, **kwargs):
        self._stage_succ = []
        self.update_metric(
            *args,
            **kwargs,
        )

    def update_metric(self, *args, task, **kwargs):
        self._metric = {}
        for stage_name, logical_expr in task.pddl_problem.stage_goals.items():
            succ_k = f"{stage_name}_success"
            if stage_name in self._stage_succ:
                self._metric[succ_k] = 1.0
            else:
                if task.pddl_problem.is_expr_true(logical_expr):
                    self._metric[succ_k] = 1.0
                    self._stage_succ.append(stage_name)
                else:
                    self._metric[succ_k] = 0.0
