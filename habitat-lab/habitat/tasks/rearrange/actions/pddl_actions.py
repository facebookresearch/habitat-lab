# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from gym import spaces

from habitat.core.registry import registry
from habitat.tasks.rearrange.actions.grip_actions import ArticulatedAgentAction


@registry.register_task_action
class PddlApplyAction(ArticulatedAgentAction):
    def __init__(self, *args, task, **kwargs):
        self._action_ordering = task.pddl_problem.get_ordered_actions()
        self._entities_list = task.pddl_problem.get_ordered_entities_list()
        super().__init__(*args, **kwargs)
        self._task = task
        self._was_prev_action_invalid = False

    @property
    def entities(self):
        return self._entities_list

    @property
    def action_space(self):
        action_n_args = sum(
            [action.n_args for action in self._action_ordering]
        )

        return spaces.Dict(
            {
                self._action_arg_prefix
                + "pddl_action": spaces.Box(
                    shape=(action_n_args,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                )
            }
        )

    @property
    def was_prev_action_invalid(self):
        return self._was_prev_action_invalid

    def reset(self, *args, **kwargs):
        self._was_prev_action_invalid = False
        self._prev_action = None

    def get_pddl_action_start(self, action_id: int) -> int:
        start_idx = 0
        for action in self._action_ordering[:action_id]:
            start_idx += action.n_args
        return start_idx

    def _apply_action(self, apply_pddl_action):
        cur_i = 0
        for action in self._action_ordering:
            action_part = apply_pddl_action[cur_i : cur_i + action.n_args][:]
            if sum(action_part) > 0:
                # Take action
                # Convert 1 indexed to 0 indexed.
                real_action_idxs = [int(a) - 1 for a in action_part]
                for a in real_action_idxs:
                    if a < 0.0:
                        raise ValueError(
                            f"Got invalid action value < 0 in {action_part} with action {action}"
                        )

                param_values = [self.entities[i] for i in real_action_idxs]

                # Look up the most recent version of this action.
                apply_action = self._task.pddl_problem.actions[
                    action.name
                ].clone()
                apply_action.set_param_values(param_values)
                self._prev_action = apply_action

                if not apply_action.apply_if_true(
                    self._task.pddl_problem.sim_info
                ):
                    self._was_prev_action_invalid = True

            cur_i += action.n_args

    def step(self, *args, **kwargs):
        self._prev_action = None
        apply_pddl_action = kwargs[self._action_arg_prefix + "pddl_action"]
        self._was_prev_action_invalid = False
        inputs_outside = any(
            a < 0 or a > len(self.entities) for a in apply_pddl_action
        )
        if not inputs_outside:
            self._apply_action(apply_pddl_action)
