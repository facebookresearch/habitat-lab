#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.rearrange.multi_task.pddl_domain import PddlProblem
from habitat.tasks.rearrange.rearrange_task import RearrangeTask


@registry.register_task(name="RearrangePddlTask-v0")
class PddlTask(RearrangeTask):
    """
    Task that sets up PDDL manager.
    """

    def __init__(self, *args, config, **kwargs):
        task_spec_path = osp.join(
            config.task_spec_base_path,
            config.task_spec + ".yaml",
        )

        self.pddl_problem = PddlProblem(
            config.pddl_domain_def,
            task_spec_path,
            config,
        )

        super().__init__(config=config, *args, **kwargs)

    def reset(self, episode: Episode):
        super().reset(episode, fetch_observations=False)
        self.pddl_problem.bind_to_instance(self._sim, self)
        self._sim.maybe_update_articulated_agent()
        return self._get_observations(episode)
