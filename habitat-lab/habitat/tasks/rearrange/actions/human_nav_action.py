# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from gym import spaces

import habitat_sim
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.rearrange.actions.actions import BaseVelAction, HumanJointAction
from habitat.tasks.rearrange.utils import get_robot_spawns
from habitat.tasks.utils import get_angle

import magnum as mn
import human_controllers.amass_human_controller as amass_human_controller

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
class HumanNavAction(HumanJointAction):
    def __init__(self, *args, task, **kwargs):
        super().__init__(*args, **kwargs)
        self._task = task
        self.human_controller = None
        self._poss_actions = get_possible_nav_to_actions(task.pddl_problem)

        self._prev_ep_id = None
        self._targets = {}
        self.gen = 0

    @property
    def action_space(self):
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "human_nav_action": spaces.Box(
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
            # start_pos = self._sim.robot.base_pos
            start_pos, _, _ = get_robot_spawns(
                np.array(obj_pos),
                0.0,
                10,
                self._sim,
                20,
                1,
            )
            # TODO: not sure if it is clean to access this here
            self.human_controller.reset(self.cur_human.sim_obj.translation)
            self._targets[nav_to_target_idx] = (start_pos, np.array(obj_pos))
        return self._targets[nav_to_target_idx]

    def _path_to_point(self, point):
        agent_pos = self.cur_human.base_pos

        path = habitat_sim.ShortestPath()
        path.requested_start = agent_pos
        path.requested_end = point
        found_path = self._sim.pathfinder.find_path(path)

        # colors = [mn.Color3.red(), mn.Color3.yellow(), mn.Color3.green()]
        # self._sim.add_gradient_trajectory_object(
        #     "current_path_{}".format(self.gen), path.points, colors=colors, radius=0.03)
        # breakpoint()
        self.gen += 1
        if not found_path:
            breakpoint()
            return [agent_pos, point]
        return path.points

    def step(self, *args, is_last_action, **kwargs):
        nav_to_target_idx = kwargs[
            self._action_arg_prefix + "human_nav_action"
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
        curr_path_points = self._path_to_point(final_nav_targ)

        index = 0
        distance = 0

        robot_pos = np.array(self.cur_human.base_pos)
        while index < (len(curr_path_points) - 1) and distance < 0.1:
            index += 1
            distance = np.linalg.norm(curr_path_points[index] - robot_pos)

        cur_nav_targ = self._path_to_point(final_nav_targ)[index]
        # breakpoint()
        # base_T = self.cur_human.base_transformation
        # forward = np.array([1.0, 0, 0])
        # robot_forward = np.array(base_T.transform_vector(forward))

        # Compute relative target.
        rel_targ = cur_nav_targ - robot_pos
        # Compute heading angle (2D calculation)
        # robot_forward = robot_forward[[0, 2]]
        # rel_targ = rel_targ[[0, 2]]
        # rel_pos = (obj_targ_pos - robot_pos)[[0, 2]]

        # angle_to_target = get_angle(robot_forward, rel_targ)
        # angle_to_obj = get_angle(robot_forward, rel_pos)

        dist_to_final_nav_targ = np.linalg.norm(
            (final_nav_targ - robot_pos)[[0, 2]]
        )
        dist_to_curr_nav_targ = np.linalg.norm(
            (cur_nav_targ - robot_pos)[[0, 2]]
        )
        print(dist_to_final_nav_targ, dist_to_curr_nav_targ)

        at_goal = (
            dist_to_final_nav_targ < self._config.dist_thresh
        )
        if not at_goal:
            new_pos, new_trans = self.human_controller.walk(rel_targ)
        else:
            new_pos, new_trans = self.human_controller.stop()
        # breakpoint()
        # print("DISTANCE AND MOTION", dist_to_final_nav_targ, rel_targ, new_trans.translation, 'offset', self.human_controller.translation_offset)
        base_action = amass_human_controller.AmassHumanController.transformAction(new_pos, new_trans)
        kwargs[f"{self._action_arg_prefix}human_joints_trans"] = base_action
        return super().step(*args, is_last_action=is_last_action, **kwargs)
