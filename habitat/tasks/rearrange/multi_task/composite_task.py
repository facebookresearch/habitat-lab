#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
from typing import Any, Dict, Optional

import numpy as np

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.rearrange.marker_info import MarkerInfo
from habitat.tasks.rearrange.multi_task.pddl_domain import PddlProblem
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.tasks.rearrange.utils import rearrange_logger


@registry.register_task(name="RearrangeCompositeTask-v0")
class CompositeTask(RearrangeTask):
    """
    All tasks using a combination of sub-tasks (skills) should utilize this task.
    """

    def __init__(self, *args, config, dataset=None, **kwargs):
        task_spec_path = osp.join(
            config.TASK_SPEC_BASE_PATH, config.TASK_SPEC + ".yaml"
        )

        self.pddl_problem = PddlProblem(
            config.PDDL_DOMAIN_DEF,
            task_spec_path,
            config,
        )

        super().__init__(config=config, *args, dataset=dataset, **kwargs)

        self._cur_node_idx: int = -1
        self._cached_tasks: Dict[str, RearrangeTask] = {}
        self._cur_state = None

        # Based on the current environment state, we can infer which subtask
        # from the solution list the agent is currently executing.
        self._inferred_cur_node_idx: int = -1
        self._inferred_cur_task: Optional[RearrangeTask] = None

    def jump_to_node(
        self, node_idx: int, episode: Episode, is_full_task: bool = False
    ) -> None:
        """
        Sequentially applies all solution actions before `node_idx`. But NOT
        including the solution action at index `node_idx`.

        :param node_idx: An integer in [0, len(solution)).
        :param is_full_task: If true, then calling reset will always the task to this solution node.
        """

        rearrange_logger.debug(
            "Jumping to node {node_idx}, is_full_task={is_full_task}"
        )
        # We don't want to reset to this node if we are in full task mode.
        if not is_full_task:
            self._cur_node_idx = node_idx

        for i in range(node_idx):
            self.pddl_problem.apply_action(self.pddl_problem.solution[i])
        if node_idx not in self._cached_tasks:
            self.pddl_problem.solution[i]
            task = self._solution[node_idx].init_task(self, episode)
            self._cached_tasks[node_idx] = task
        else:
            self._cached_tasks[node_idx].reset(episode)

    def reset(self, episode: Episode):
        super().reset(episode, fetch_observations=False)
        self.pddl_problem.bind_to_instance(
            self._sim, self._dataset, self, episode
        )

        if self._cur_node_idx >= 0:
            self.jump_to_node(self._cur_node_idx, episode)

        self._inferred_cur_node_idx = 0
        self._inferred_cur_task = None
        if self._config.USING_SUBTASKS:
            self._increment_solution_subtask(episode)
        self._cached_tasks.clear()
        return self._get_observations(episode)

    def get_inferred_node_idx(self) -> int:
        if self._cur_node_idx >= 0:
            return self._cur_node_idx
        if not self._config.USING_SUBTASKS:
            raise ValueError(
                "Cannot get inferred sub-task when task is not configured to use sub-tasks. See `TASK.USING_SUBTASKS` key."
            )
        return self._inferred_cur_node_idx

    def get_inferred_node_task(self) -> RearrangeTask:
        if self._cur_node_idx >= 0:
            return self._cached_tasks[self._cur_node_idx]

        if not self._config.USING_SUBTASKS:
            raise ValueError(
                "Cannot get inferred sub-task when task is not configured to use sub-tasks. See `TASK.USING_SUBTASKS` key."
            )
        return self._inferred_cur_task

    def increment_inferred_solution_idx(self, episode: Episode) -> None:
        """
        Increment to the next index in the solution list. If the solution is
        exhausted then stay at the last index. This will update both
        `inferred_node_idx` and `inferrred_cur_task`.
        """
        prev_inf_cur_node = self._inferred_cur_node_idx
        self._inferred_cur_node_idx += 1
        if not self._increment_solution_subtask(episode):
            self._inferred_cur_node_idx = prev_inf_cur_node

    def _increment_solution_subtask(self, episode: Episode) -> bool:
        """
        Gets the next inferred sub-task in the solution list. Returns False if
        there are no remaining sub-tasks in the solution list.
        """
        task_solution = self.solution
        if self._inferred_cur_node_idx >= len(task_solution):
            return False
        self._inferred_cur_node_idx += 1
        if self._inferred_cur_node_idx >= len(task_solution):
            return False

        prev_state = self._sim.capture_state(with_robot_js=True)
        if self._inferred_cur_node_idx in self._cached_tasks:
            self._inferred_cur_task = self._cached_tasks[
                self._inferred_cur_node_idx
            ]
            self._inferred_cur_task.reset(episode)
            rearrange_logger.debug(
                f"Incrementing solution to {self._inferred_cur_node_idx}. Loading next task from cached"
            )
        else:
            rearrange_logger.debug(
                f"Incrementing solution to {self._inferred_cur_node_idx}. Loading next task."
            )
            task = task_solution[self._inferred_cur_node_idx].init_task(
                self.pddl_problem.sim_info, should_reset=False
            )
            self._cached_tasks[self._inferred_cur_node_idx] = task
            self._inferred_cur_task = task
        self._sim.set_state(prev_state)

        return True

    def _try_get_subtask_prop(self, prop_name: str, def_val: Any) -> Any:
        """
        Try to get a property from the current inferred subtask. If the subtask
        is not valid, then return the supplied default value.
        """
        inferred_task = self.get_inferred_node_task()
        if inferred_task is not None and hasattr(inferred_task, prop_name):
            return getattr(inferred_task, prop_name)
        return def_val

    #########################################################################
    # START Sub-task property overrides
    # These will emulate properties from sub-tasks needed to compute sub-task
    # sensors and measurements.
    #########################################################################

    @property
    def targ_idx(self) -> int:
        return self._try_get_subtask_prop("targ_idx", self._targ_idx)

    @property
    def abs_targ_idx(self) -> int:
        if self._targ_idx is None:
            abs_targ_idx = None
        else:
            abs_targ_idx = self._sim.get_targets()[0][self._targ_idx]

        return self._try_get_subtask_prop("abs_targ_idx", abs_targ_idx)

    @property
    def nav_to_task_name(self) -> str:
        return self._try_get_subtask_prop("nav_to_task_name", None)

    @property
    def nav_to_entity_name(self) -> str:
        return self._try_get_subtask_prop("nav_to_entity_name", "")

    @property
    def nav_target_pos(self) -> np.ndarray:
        return self._try_get_subtask_prop("nav_target_pos", np.zeros((3,)))

    @property
    def nav_target_angle(self) -> float:
        return self._try_get_subtask_prop("nav_target_angle", 0.0)

    @property
    def success_js_state(self) -> float:
        return 0.0

    @property
    def use_marker_name(self) -> str:
        subtask_marker_name = self._try_get_subtask_prop(
            "use_marker_name", None
        )
        if subtask_marker_name is not None:
            return subtask_marker_name
        else:
            all_markers = self._sim.get_all_markers()
            return list(all_markers.keys())[0]

    def get_use_marker(self) -> MarkerInfo:
        return self._sim.get_marker(self.use_marker_name)

    #########################################################################
    # END Sub-task property overrides
    #########################################################################
