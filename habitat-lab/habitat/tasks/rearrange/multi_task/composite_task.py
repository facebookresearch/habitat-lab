#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
from typing import cast

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.datasets.rearrange.rearrange_dataset import RearrangeDatasetV0
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
            config.task_spec_base_path,
            config.task_spec + ".yaml",
        )

        self.pddl_problem = PddlProblem(
            config.pddl_domain_def,
            task_spec_path,
            config,
        )

        super().__init__(config=config, *args, dataset=dataset, **kwargs)

        self._cur_node_idx: int = -1

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

    def reset(self, episode: Episode):
        super().reset(episode, fetch_observations=False)
        self.pddl_problem.bind_to_instance(
            self._sim, cast(RearrangeDatasetV0, self._dataset), self, episode
        )

        if self._cur_node_idx >= 0:
            self.jump_to_node(self._cur_node_idx, episode)

        self._sim.maybe_update_robot()
        return self._get_observations(episode)
