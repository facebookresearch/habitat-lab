# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import magnum as mn
import numpy as np
from gym import spaces

import habitat_sim
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.rearrange.actions.actions import (
    BaseVelAction,
    HumanoidJointAction,
)
from habitat.tasks.rearrange.utils import get_robot_spawns
from habitat.tasks.utils import get_angle


@registry.register_task_action
class OracleNavAction(BaseVelAction, HumanoidJointAction):
    """
    An action that will convert the index of an entity (in the sense of
    `PddlEntity`) to navigate to and convert this to base/humanoid joint control to move the
    robot to the closest navigable position to that entity. The entity index is
    the index into the list of all available entities in the current scene. The
    config flag motion_type indicates whether the low level action will be a base_velocity or
    a joint control.
    """

    def __init__(self, *args, task, **kwargs):
        config = kwargs["config"]
        self.motion_type = config.motion_control
        if self.motion_type == "base_velocity":
            BaseVelAction.__init__(self, *args, **kwargs)
        else:
            self.humanoid_controller = None
            HumanoidJointAction.__init__(self, *args, **kwargs)
        self._task = task
        self._poss_entities = (
            self._task.pddl_problem.get_ordered_entities_list()
        )
        self._prev_ep_id = None
        self._targets = {}

        self.curr_ind_map = {"cont": 0}

    @staticmethod
    def _compute_turn(rel, turn_vel, robot_forward):
        is_left = np.cross(robot_forward, rel) > 0
        if is_left:
            vel = [0, -turn_vel]
        else:
            vel = [0, turn_vel]
        return vel

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

        self.curr_ind_map = {"cont": self.curr_ind_map["cont"] + 1}

    def _get_target_for_idx(self, nav_to_target_idx: int):
        if nav_to_target_idx not in self._targets:
            nav_to_obj = self._poss_entities[nav_to_target_idx]
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
            if self.motion_type == "human_joints":
                self.humanoid_controller.reset(
                    self.cur_articulated_agent.base_pos
                )
            self._targets[nav_to_target_idx] = (start_pos, np.array(obj_pos))
        return self._targets[nav_to_target_idx]

    def _path_to_point(self, point, agent_pos=None):
        if agent_pos is None:
            agent_pos = self.cur_articulated_agent.base_pos

        path = habitat_sim.ShortestPath()
        path.requested_start = agent_pos
        path.requested_end = point
        found_path = self._sim.pathfinder.find_path(path)
        if not found_path:
            return [agent_pos, point]
        return path.points

    def step(self, *args, is_last_action, **kwargs):
        self.humanoid_controller._sim = self._sim
        nav_to_target_idx = kwargs[
            self._action_arg_prefix + "oracle_nav_action"
        ]
        if nav_to_target_idx <= 0 or nav_to_target_idx > len(
            self._poss_entities
        ):
            if is_last_action:
                return self._sim.step(HabitatSimActions.base_velocity)
            else:
                return {}
        nav_to_target_idx = int(nav_to_target_idx[0]) - 1

        final_nav_targ, obj_targ_pos = self._get_target_for_idx(
            nav_to_target_idx
        )
        if self.motion_type != "base_velocity":
            base_T = self.cur_articulated_agent.base_transformation
            base_pos, _ = self.humanoid_controller.get_corrected_base(base_T)
            bpos = self.cur_articulated_agent.base_transformation.translation
            # print("CORRECTED", base_pos, "BASEPOS", bpos)
            # breakpoint()
            curr_path_points = self._path_to_point(final_nav_targ, base_pos)
        else:
            curr_path_points = self._path_to_point(final_nav_targ)

        colors = [mn.Color3.red(), mn.Color3.yellow(), mn.Color3.green()]

        if "trajectory" in self.curr_ind_map:
            # pass
            # print(self.curr_ind_map)
            self._sim.get_rigid_object_manager().remove_object_by_id(
                self.curr_ind_map["trajectory"]
            )
        self.curr_ind_map[
            "trajectory"
        ] = self._sim.add_gradient_trajectory_object(
            "current_path_{}".format(self.curr_ind_map["cont"]),
            curr_path_points,
            colors=colors,
            radius=0.03,
        )
        self.curr_ind_map["cont"] += 1

        if curr_path_points is None:
            raise Exception
        else:
            cur_nav_targ = curr_path_points[1]

            robot_pos = np.array(self.cur_articulated_agent.base_pos)
            base_T = self.cur_articulated_agent.base_transformation

            if self.motion_type != "base_velocity":
                # Need to correct base and robot_pos
                # print(base_T)
                (
                    robot_pos2,
                    base_T2,
                ) = self.humanoid_controller.get_corrected_base(base_T)
                # breakpoint()
                forward = np.array([0.0, 0, 1.0])
                fw1 = np.array(base_T.transform_vector(forward))
                fw2 = np.array(base_T2.transform_vector(-forward))
                self.humanoid_controller.gen_map(robot_pos, robot_pos + fw1)
                self.humanoid_controller.gen_map(
                    robot_pos2, robot_pos2 + fw2, c=2
                )
                print(robot_pos, robot_pos2)
                robot_pos, base_T = robot_pos2, base_T2

            forward = np.array([1.0, 0, 0])
            robot_forward = np.array(base_T.transform_vector(forward))

            # Compute relative target.
            rel_targ = cur_nav_targ - robot_pos
            # Compute heading angle (2D calculation)
            robot_forward = robot_forward[[0, 2]]
            rel_targ = rel_targ[[0, 2]]
            rel_pos = (obj_targ_pos - robot_pos)[[0, 2]]

            vec1 = mn.Vector3([rel_targ[0], 0.0, rel_targ[1]])
            # self.humanoid_controller.gen_map(robot_pos, robot_pos + vec1, c=2)

            angle_to_target = get_angle(robot_forward, rel_targ)
            angle_to_obj = get_angle(robot_forward, rel_pos)

            dist_to_final_nav_targ = np.linalg.norm(
                (final_nav_targ - robot_pos)[[0, 2]]
            )
            at_goal = (
                dist_to_final_nav_targ < self._config.dist_thresh
                and angle_to_obj < self._config.turn_thresh
            )
            if self.humanoid_controller.rotate_count > 30:
                print("NAV TARGET", dist_to_final_nav_targ, angle_to_obj)
                print("", self._config.dist_thresh, self._config.turn_thresh)
            if self.motion_type == "base_velocity":
                if not at_goal:
                    if dist_to_final_nav_targ < self._config.dist_thresh:
                        # Look at the object
                        vel = OracleNavAction._compute_turn(
                            rel_pos, self._config.turn_velocity, robot_forward
                        )
                    elif angle_to_target < self._config.turn_thresh:
                        # Move towards the target
                        vel = [self._config.forward_velocity, 0]
                    else:
                        # Look at the target waypoint.
                        vel = OracleNavAction._compute_turn(
                            rel_targ, self._config.turn_velocity, robot_forward
                        )
                else:
                    vel = [0, 0]
                kwargs[f"{self._action_arg_prefix}base_vel"] = np.array(vel)
                return BaseVelAction.step(
                    self, *args, is_last_action=is_last_action, **kwargs
                )

            else:
                if not at_goal:
                    if dist_to_final_nav_targ < self._config.dist_thresh:
                        print("TURN to obj!!")
                        # Look at the object
                        (
                            new_pos,
                            new_trans,
                        ) = self.humanoid_controller.compute_turn(
                            mn.Vector3([rel_pos[0], 0.0, rel_pos[1]])
                        )
                    else:
                        # Move towards the target
                        (
                            new_pos,
                            new_trans,
                        ) = self.humanoid_controller.get_walk_pose(
                            mn.Vector3([rel_targ[0], 0.0, rel_targ[1]])
                        )
                else:
                    (
                        new_pos,
                        new_trans,
                    ) = self.humanoid_controller.get_stop_pose()
                if new_trans is None:
                    breakpoint()
                base_action = self.humanoid_controller.vectorize_pose(
                    new_pos, new_trans
                )
                kwargs[
                    f"{self._action_arg_prefix}human_joints_trans"
                ] = base_action

                return HumanoidJointAction.step(
                    self, *args, is_last_action=is_last_action, **kwargs
                )
