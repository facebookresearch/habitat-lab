# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from gym import spaces

from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.rearrange.actions.actions import HumanoidJointAction
from habitat.tasks.rearrange.actions.oracle_nav_action import OracleNavAction
from habitat.tasks.utils import get_angle
from habitat_baselines.articulated_agent_controller.humanoid_rearrange_controller import (
    HumanoidRearrangeController,
)


@registry.register_task_action
class OracleNavHumanoidAction(OracleNavAction, HumanoidJointAction):
    """
    An action that will convert the index of an entity (in the sense of
    `PddlEntity`) to navigate to and convert this to base control to move the
    robot to the closest navigable position to that entity. The entity index is
    the index into the list of all available entities in the current scene.
    """

    def __init__(self, *args, task, **kwargs):
        HumanoidJointAction.__init__(*args, **kwargs)

        self.humanoid_controller = None
        self._task = task

        self._poss_entities = (
            self._task.pddl_problem.get_ordered_entities_list()
        )
        self._prev_ep_id = None
        self._targets = {}

    @property
    def action_space(self):
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "oracle_nav_humanoid_action": spaces.Box(
                    shape=(1,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                )
            }
        )

    def _get_target_for_idx(self, nav_to_target_idx: int):
        super()._get_target_for_idx(nav_to_target_idx)
        self.humanoid_controller.reset(self.cur_articulated_agent.base_pos)
        return self._targets[nav_to_target_idx]

    def step(self, *args, is_last_action, **kwargs):
        nav_to_target_idx = kwargs[
            self._action_arg_prefix + "oracle_nav_humanoid_action"
        ]
        if nav_to_target_idx <= 0 or nav_to_target_idx > len(
            self._poss_entities
        ):
            if is_last_action:
                return self._sim.step(HabitatSimActions.changejoint_action)
            else:
                return {}
        nav_to_target_idx = int(nav_to_target_idx[0]) - 1

        final_nav_targ, obj_targ_pos = self._get_target_for_idx(
            nav_to_target_idx
        )

        curr_path_points = self._path_to_point(final_nav_targ)

        if curr_path_points is None:
            # path not found
            new_pos, new_trans = self.humanoid_controller.stop()
        else:
            cur_nav_targ = curr_path_points[1]

            robot_pos = np.array(self.cur_articulated_agent.base_pos)
            base_T = self.cur_articulated_agent.base_transformation

            forward = np.array([1.0, 0, 0])
            robot_forward = np.array(base_T.transform_vector(forward))

            # Compute relative target.
            rel_targ = cur_nav_targ - robot_pos
            # Compute heading angle (2D calculation)
            robot_forward = robot_forward[[0, 2]]
            rel_targ = rel_targ[[0, 2]]
            rel_pos = (obj_targ_pos - robot_pos)[[0, 2]]

            # angle_to_target = get_angle(robot_forward, rel_targ)
            angle_to_obj = get_angle(robot_forward, rel_pos)

            dist_to_final_nav_targ = np.linalg.norm(
                (final_nav_targ - robot_pos)[[0, 2]]
            )
            at_goal = (
                dist_to_final_nav_targ < self._config.dist_thresh
                and angle_to_obj < self._config.turn_thresh
            )

            if not at_goal:
                if dist_to_final_nav_targ < self._config.dist_thresh:
                    # Look at the object
                    new_pos, new_trans = self.humanoid_controller.compute_turn(
                        rel_pos
                    )
                else:
                    # Move towards the target
                    new_pos, new_trans = self.humanoid_controller.walk(
                        rel_targ
                    )

            else:
                new_pos, new_trans = self.humanoid_controller.stop()

        base_action = HumanoidRearrangeController.VectorizePose(
            new_pos, new_trans
        )

        kwargs[f"{self._action_arg_prefix}changejoints_trans"] = base_action
        return HumanoidJointAction.step(
            *args, is_last_action=is_last_action, **kwargs
        )
