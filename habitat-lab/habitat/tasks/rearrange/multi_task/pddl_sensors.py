#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import List

import numpy as np
from gym import spaces

from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.rearrange.multi_task.pddl_task import PddlTask
from habitat.tasks.rearrange.rearrange_sensors import (
    DoesWantTerminate,
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
        assert isinstance(task, PddlTask)
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
            shape=(len(self.predicates_list),), low=0, high=1, dtype=np.float32
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
        super().__init__(*args, **kwargs)

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                ObjectToGoalDistance.cls_uuid,
                EndEffectorToObjectDistance.cls_uuid,
            ],
        )

        self._gave_pick_reward = {}
        self._prev_holding_obj = False
        self.num_targets = len(self._sim.get_targets()[0])

        self._cur_rearrange_stage = 0
        self.update_target_object()

        self._prev_obj_to_goal_dist = self.get_distance(
            task, ObjectToGoalDistance
        )
        self._prev_ee_to_obj_dist = self.get_distance(
            task, EndEffectorToObjectDistance
        )

        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_target_object(self):
        """
        The agent just finished one rearrangement stage so it's time to
        update the target object for the next stage.
        """
        # Get the next target object
        idxs, _ = self._sim.get_targets()
        targ_obj_idx = idxs[self._cur_rearrange_stage]

        # Get the target object's absolute index
        self.abs_targ_obj_idx = self._sim.scene_obj_ids[targ_obj_idx]

    def get_distance(self, task, distance):
        return task.measurements.measures[distance.cls_uuid].get_metric()[
            str(self._cur_rearrange_stage)
        ]

    def update_metric(self, *args, episode, task, observations, **kwargs):
        super().update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )
        # If all the objects are in the right place but we haven't succeded in the task
        # (for example the agent hasn't called terminate) we give zero reward

        if self._cur_rearrange_stage == self.num_targets:
            self._metric = 0
            return

        obj_to_goal_dist = self.get_distance(task, ObjectToGoalDistance)
        ee_to_obj_dist = self.get_distance(task, EndEffectorToObjectDistance)

        is_holding_obj = self._sim.grasp_mgr.snap_idx == self.abs_targ_obj_idx
        picked_up_obj = is_holding_obj and not self._prev_holding_obj

        # DISTANCE REWARD: Steers the agent towards the object and then towards the goal

        if is_holding_obj:
            dist_diff = self._prev_obj_to_goal_dist - obj_to_goal_dist
        else:
            dist_diff = self._prev_ee_to_obj_dist - ee_to_obj_dist
        self._metric += self._config.dist_reward * dist_diff

        # PICK REWARD: Reward for picking up the object, only given once to avoid
        # reward hacking.

        already_gave_reward = (
            self._cur_rearrange_stage in self._gave_pick_reward
        )
        if picked_up_obj and not already_gave_reward:
            self._metric += self._config.pick_reward
            self._gave_pick_reward[self._cur_rearrange_stage] = True

        # PLACE REWARD: Reward for placing the object correcly (within success dist)
        # We also udate the target object for the next stage.

        place_success = obj_to_goal_dist < self._config.success_dist
        if place_success and not is_holding_obj:
            self._metric += self._config.single_rearrange_reward
            self._cur_rearrange_stage += 1
            self._cur_rearrange_stage = (
                self._cur_rearrange_stage % self.num_targets
            )
            if self._cur_rearrange_stage < self.num_targets:
                self.update_target_object()

        # Need to call the get_distance functions again because the target
        # object may have changed in the previous if statement.
        self._prev_obj_to_goal_dist = self.get_distance(
            task, ObjectToGoalDistance
        )
        self._prev_ee_to_obj_dist = self.get_distance(
            task, EndEffectorToObjectDistance
        )
        self._prev_holding_obj = is_holding_obj


@registry.register_measure
class PddlSuccess(Measure):
    """
    Did satisfy all the goal predicates?
    """

    cls_uuid: str = "pddl_success"

    def __init__(self, sim, config, *args, task, **kwargs):
        super().__init__(**kwargs)
        self._sim = sim
        self._config = config

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return PddlSuccess.cls_uuid

    def reset_metric(self, *args, task, **kwargs):
        if self._config.must_call_stop:
            task.measurements.check_measure_dependencies(
                self.uuid, [DoesWantTerminate.cls_uuid]
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
class PddlStageGoals(Measure):
    """
    Adds to the metrics `[TASK_NAME]_success`: Did the agent complete a
        particular stage defined in `stage_goals` at ANY point in the episode.
    """

    _stage_succ: List[str]
    cls_uuid: str = "pddl_stage_goals"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return PddlStageGoals.cls_uuid

    def reset_metric(self, *args, **kwargs):
        self._stage_succ = []
        self.update_metric(*args, **kwargs)

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


@registry.register_measure
class PddlSubgoalReward(Measure):
    """
    Reward that gives a sparse reward on completing a PDDL stage-goal.
    """

    cls_uuid: str = "pddl_subgoal_reward"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return PddlSubgoalReward.cls_uuid

    def __init__(self, *args, task, config, **kwargs):
        assert isinstance(task, PddlTask)
        super().__init__(*args, task=task, config=config, **kwargs)
        self._stage_reward = config.stage_sparse_reward

    def reset_metric(self, *args, **kwargs):
        self._stage_succ = []
        self.update_metric(
            *args,
            **kwargs,
        )

    def _get_stage_reward(self, name):
        return self._stage_reward

    def update_metric(self, *args, task, **kwargs):
        self._metric = 0.0

        for stage_name, logical_expr in task.pddl_problem.stage_goals.items():
            if stage_name in self._stage_succ:
                continue

            if task.pddl_problem.is_expr_true(logical_expr):
                self._metric += self._get_stage_reward(stage_name)
                self._stage_succ.append(stage_name)
