# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from gym import spaces

import habitat_sim
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.rearrange.actions.actions import BaseVelAction
from habitat.tasks.rearrange.utils import get_robot_spawns
from habitat.tasks.utils import get_angle


def compute_turn(rel, turn_vel, robot_forward):
    is_left = np.cross(robot_forward, rel) > 0
    if is_left:
        vel = [0, -turn_vel]
    else:
        vel = [0, turn_vel]
    return vel


def get_possible_nav_to_actions(pddl_problem):
    return pddl_problem.get_possible_actions(
        allowed_action_names=["nav", "nav_to_receptacle"],
        true_preds=None,
    )


@registry.register_task_action
class OracleNavAction(BaseVelAction):
    def __init__(self, *args, task, **kwargs):
        super().__init__(*args, **kwargs)
        self._task = task

        self._poss_actions = get_possible_nav_to_actions(task.pddl_problem)
        self._prev_ep_id = None
        self._targets = {}

    @property
    def action_space(self):
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "oracle_nav_action": spaces.Box(
                    shape=(1,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                )
            }
        )

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        if self._task._episode_id != self._prev_ep_id:
            self._targets = {}
            self._prev_ep_id = self._task._episode_id

    def _get_target_for_idx(self, nav_to_target_idx: int):
        if nav_to_target_idx not in self._targets:
            action = self._poss_actions[nav_to_target_idx]
            nav_to_obj = action.get_arg_value("obj")
            obj_pos = self._task.pddl_problem.sim_info.get_entity_pos(
                nav_to_obj
            )
            start_pos, _, _ = get_robot_spawns(
                np.array(obj_pos),
                0.0,
                self._config.spawn_max_dist_to_obj,
                self._sim,
                self._config.num_spawn_attempts,
                1,
            )

            self._targets[nav_to_target_idx] = (start_pos, np.array(obj_pos))
        return self._targets[nav_to_target_idx]

    def _path_to_point(self, point):
        agent_pos = self.cur_robot.base_pos

        path = habitat_sim.ShortestPath()
        path.requested_start = agent_pos
        path.requested_end = point
        found_path = self._sim.pathfinder.find_path(path)
        if not found_path:
            return [agent_pos, point]
        return path.points

    def step(self, *args, is_last_action, **kwargs):
        nav_to_target_idx = kwargs[
            self._action_arg_prefix + "oracle_nav_action"
        ]
        if nav_to_target_idx <= 0 or nav_to_target_idx > len(
            self._poss_actions
        ):
            if is_last_action:
                return self._sim.step(HabitatSimActions.base_velocity)
            else:
                return {}
        nav_to_target_idx = int(nav_to_target_idx[0]) - 1

        final_nav_targ, obj_targ_pos = self._get_target_for_idx(
            nav_to_target_idx
        )
        cur_nav_targ = self._path_to_point(final_nav_targ)[1]

        robot_pos = np.array(self.cur_robot.base_pos)
        base_T = self.cur_robot.base_transformation
        forward = np.array([1.0, 0, 0])
        robot_forward = np.array(base_T.transform_vector(forward))

        # Compute relative target.
        rel_targ = cur_nav_targ - robot_pos
        # Compute heading angle (2D calculation)
        robot_forward = robot_forward[[0, 2]]
        rel_targ = rel_targ[[0, 2]]
        rel_pos = (obj_targ_pos - robot_pos)[[0, 2]]

        angle_to_target = get_angle(robot_forward, rel_targ)
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
                vel = compute_turn(
                    rel_pos, self._config.turn_velocity, robot_forward
                )
            elif angle_to_target < self._config.turn_thresh:
                # Move towards the target
                vel = [self._config.forward_velocity, 0]
            else:
                # Look at the target waypoint.
                vel = compute_turn(
                    rel_targ, self._config.turn_velocity, robot_forward
                )
        else:
            vel = [0, 0]

        kwargs[f"{self._action_arg_prefix}base_vel"] = np.array(vel)
        return super().step(*args, is_last_action=is_last_action, **kwargs)
