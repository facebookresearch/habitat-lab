#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.tasks.rearrange.utils import logger


@registry.register_measure
class CompositeReward(Measure):
    """
    The reward based on where the agent currently is in the hand defined solution list.
    """

    cls_uuid: str = "composite_reward"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return CompositeReward.cls_uuid

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(**kwargs)
        self._sim = sim
        self._config = config
        self._prev_node_idx = None

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [CompositeNodeIdx.cls_uuid],
        )

        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        reward = 0.0
        node_measure = task.measurements.measures[CompositeNodeIdx.cls_uuid]

        node_idx = node_measure.get_metric()["node_idx"]
        if self._prev_node_idx is None:
            self._prev_node_idx = node_idx
        elif node_idx > self._prev_node_idx:
            reward += self._config.STAGE_COMPLETE_REWARD

        cur_task = task.get_cur_task()
        if cur_task is None:
            cur_task_cfg = task.get_inferrred_node_task()._config
        else:
            cur_task_cfg = cur_task._config

        if cur_task_cfg.REWARD_MEASUREMENT == "":
            raise ValueError("REWARD_MEASUREMENT key not defined")
        cur_task_reward = task.measurements.measures[
            cur_task_cfg.REWARD_MEASUREMENT
        ].get_metric()
        reward += cur_task_reward

        self._metric = cur_task_reward
        self._prev_node_idx = node_idx


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

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        if task.get_cur_task() is not None:
            # Don't check success when we are evaluating a subtask.
            self._metric = False
        else:
            self._metric = task.is_goal_state_satisfied()


@registry.register_measure
class CompositeNodeIdx(Measure):
    """
    Adds several keys to the metrics dictionary:
        - `reached_i`: Did the agent succeed in sub-task at index `i` of the
          sub-task `solution` list?
        - `node_idx`: Index of the agent in completing the sub-tasks from
          the `solution` list.
        - `[TASK_NAME]_success`: Did the agent complete a particular stage
          defined in `stage_goals`.
    """

    cls_uuid: str = "composite_node_idx"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(**kwargs)
        self._sim = sim
        self._config = config
        self._stage_succ = []

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return CompositeNodeIdx.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self._stage_succ = []
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        cur_task = task.get_cur_task()
        self._metric = {}
        if cur_task is None:
            inf_cur_task_cfg = task.get_inferrred_node_task()._config
            if inf_cur_task_cfg.SUCCESS_MEASUREMENT == "":
                raise ValueError("SUCCESS_MEASUREMENT not defined")

            is_succ = task.measurements.measures[
                inf_cur_task_cfg.SUCCESS_MEASUREMENT
            ].get_metric()
            if is_succ:
                task.increment_inferred_solution_idx(episode)
                logger.info(
                    f"Completed {inf_cur_task_cfg.TYPE}, incremented node to {task.get_inferrred_node_task()}"
                )

            node_idx = task.get_inferred_node_idx()
            for i in range(task.get_num_nodes()):
                self._metric[f"reached_{i}"] = (
                    task.get_inferred_node_idx() >= i
                )
        else:
            node_idx = task.get_cur_node()
        self._metric["node_idx"] = node_idx
        self._update_info_stage_succ(task, self._metric)

    def _update_info_stage_succ(self, task, info):
        stage_goals = task.get_stage_goals()
        for k, preds in stage_goals.items():
            succ_k = f"{k}_success"
            if k in self._stage_succ:
                info[succ_k] = 1.0
            else:
                if task.is_pred_list_sat(preds):
                    info[succ_k] = 1.0
                    self._stage_succ.append(k)
                else:
                    info[succ_k] = 0.0
