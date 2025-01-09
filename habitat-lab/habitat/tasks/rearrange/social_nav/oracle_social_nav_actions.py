# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import magnum as mn
import numpy as np
from gym import spaces

import habitat_sim
from habitat.core.registry import registry
from habitat.datasets.rearrange.navmesh_utils import (
    SimpleVelocityControlEnv,
    compute_turn,
)
from habitat.tasks.rearrange.actions.actions import (
    BaseVelAction,
    HumanoidJointAction,
)
from habitat.tasks.rearrange.actions.oracle_nav_action import OracleNavAction
from habitat.tasks.rearrange.social_nav.utils import (
    robot_human_vec_dot_product,
)
from habitat.tasks.rearrange.utils import place_agent_at_dist_from_pos
from habitat.tasks.utils import get_angle


@registry.register_task_action
class OracleNavCoordAction(OracleNavAction):  # type: ignore
    """
    An action that comments the agent to navigate to a sequence of random navigation targets (or we call these targets (x,y) coordinates)
    """

    def __init__(self, *args, task, **kwargs):
        super().__init__(*args, task=task, **kwargs)
        self.nav_mode = None
        self.simple_backward = False

    @property
    def action_space(self):
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "oracle_nav_coord_action": spaces.Box(
                    shape=(3,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                )
            }
        )

    def _get_target_for_coord(self, obj_pos):
        """Get the targets by recording them in the dict"""
        precision = 0.25
        pos_key = np.around(obj_pos / precision, decimals=0) * precision
        pos_key = tuple(pos_key)
        if pos_key not in self._targets:
            start_pos, _, _ = place_agent_at_dist_from_pos(
                np.array(obj_pos),
                0.0,
                self._config.spawn_max_dist_to_obj,
                self._sim,
                self._config.num_spawn_attempts,
                True,
                self.cur_articulated_agent,
            )
            self._targets[pos_key] = start_pos
        else:
            start_pos = self._targets[pos_key]
        if self.motion_type == "human_joints":
            self.humanoid_controller.reset(
                self.cur_articulated_agent.base_transformation
            )
        return (np.array(start_pos), np.array(obj_pos))

    def step(self, *args, **kwargs):
        self.skill_done = False
        nav_to_target_coord = kwargs.get(
            self._action_arg_prefix + "oracle_nav_coord_action",
            self._action_arg_prefix + "oracle_nav_human_action",
        )
        if np.linalg.norm(nav_to_target_coord) == 0:
            return {}
        final_nav_targ, obj_targ_pos = self._get_target_for_coord(
            nav_to_target_coord
        )

        base_T = self.cur_articulated_agent.base_transformation
        curr_path_points = self._path_to_point(final_nav_targ)
        robot_pos = np.array(self.cur_articulated_agent.base_pos)
        if curr_path_points is None:
            raise Exception
        else:
            # Compute distance and angle to target
            if len(curr_path_points) == 1:
                curr_path_points += curr_path_points
            cur_nav_targ = np.array(curr_path_points[1])
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
            ) or dist_to_final_nav_targ < self._config.dist_thresh / 10.0
            if self.motion_type == "base_velocity":
                if not at_goal:
                    if self.nav_mode == "avoid":
                        backward = np.array([-1.0, 0, 0])
                        robot_backward = np.array(
                            base_T.transform_vector(backward)
                        )
                        robot_backward = robot_backward[[0, 2]]
                        angle_to_target = get_angle(robot_backward, rel_targ)
                        if (
                            self.simple_backward
                            or angle_to_target < self._config.turn_thresh
                        ):
                            # Move backwards the target
                            vel = [self._config.forward_velocity, 0]
                        else:
                            # Robot's rear looks at the target waypoint.
                            vel = compute_turn(
                                rel_targ,
                                self._config.turn_velocity,
                                robot_backward,
                            )
                    else:
                        if dist_to_final_nav_targ < self._config.dist_thresh:
                            # Look at the object
                            vel = compute_turn(
                                rel_pos,
                                self._config.turn_velocity,
                                robot_forward,
                            )
                        elif angle_to_target < self._config.turn_thresh:
                            # Move towards the target
                            vel = [self._config.forward_velocity, 0]
                        else:
                            # Look at the target waypoint.
                            vel = compute_turn(
                                rel_targ,
                                self._config.turn_velocity,
                                robot_forward,
                            )
                else:
                    vel = [0, 0]
                    self.skill_done = True
                kwargs[f"{self._action_arg_prefix}base_vel"] = np.array(vel)
                return BaseVelAction.step(self, *args, **kwargs)

            elif self.motion_type == "human_joints":
                # Update the humanoid base
                self.humanoid_controller.obj_transform_base = base_T
                if not at_goal:
                    if dist_to_final_nav_targ < self._config.dist_thresh:
                        # Look at the object
                        self.humanoid_controller.calculate_turn_pose(
                            mn.Vector3([rel_pos[0], 0.0, rel_pos[1]])
                        )
                    else:
                        # Move towards the target
                        if self._config["lin_speed"] == 0:
                            distance_multiplier = 0.0
                        else:
                            distance_multiplier = 1.0
                        self.humanoid_controller.calculate_walk_pose(
                            mn.Vector3([rel_targ[0], 0.0, rel_targ[1]]),
                            distance_multiplier,
                        )
                else:
                    self.humanoid_controller.calculate_stop_pose()
                    self.skill_done = True
                # This line is important to reset the controller
                self._update_controller_to_navmesh()
                base_action = self.humanoid_controller.get_pose()
                kwargs[
                    f"{self._action_arg_prefix}human_joints_trans"
                ] = base_action

                return HumanoidJointAction.step(self, *args, **kwargs)
            else:
                raise ValueError(
                    "Unrecognized motion type for oracle nav action"
                )


@registry.register_task_action
class OracleNavRandCoordAction(OracleNavCoordAction):  # type: ignore
    """
    Oracle Nav RandCoord Action. Selects a random position in the scene and navigates
    there until reaching. When the arg is 1, does replanning.
    """

    def __init__(self, *args, task, **kwargs):
        super().__init__(*args, task=task, **kwargs)
        self._config = kwargs["config"]

    @property
    def action_space(self):
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "oracle_nav_randcoord_action": spaces.Box(
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
        self.skill_done = False
        self.coord_nav = None

    def _find_path_given_start_end(self, start, end):
        """Helper function to find the path given starting and end locations"""
        path = habitat_sim.ShortestPath()
        path.requested_start = start
        path.requested_end = end
        found_path = self._sim.pathfinder.find_path(path)
        if not found_path:
            return [start, end]
        return path.points

    def _reach_human(self, robot_pos, human_pos, base_T):
        """Check if the agent reaches the human or not"""
        facing = (
            robot_human_vec_dot_product(robot_pos, human_pos, base_T) > 0.5
        )

        # Use geodesic distance here
        dis = self._sim.geodesic_distance(robot_pos, human_pos)

        return dis <= 2.0 and facing

    def _compute_robot_to_human_min_step(
        self, robot_trans, human_pos, human_pos_list
    ):
        """The function to compute the minimum step to reach the goal"""
        _vel_scale = self._config.lin_speed

        # Copy the robot transformation
        base_T = mn.Matrix4(robot_trans)

        vc = SimpleVelocityControlEnv()

        # Compute the step taken to reach the human
        robot_pos = np.array(base_T.translation)
        robot_pos[1] = human_pos[1]
        step_taken = 0
        while (
            not self._reach_human(robot_pos, human_pos, base_T)
            and step_taken <= 1500
        ):
            path_points = self._find_path_given_start_end(robot_pos, human_pos)
            cur_nav_targ = path_points[1]
            obj_targ_pos = path_points[1]
            forward = np.array([1.0, 0, 0])
            robot_forward = np.array(base_T.transform_vector(forward))

            # Compute relative target.
            rel_targ = cur_nav_targ - robot_pos
            rel_pos = (obj_targ_pos - robot_pos)[[0, 2]]

            # Compute heading angle (2D calculation)
            robot_forward = robot_forward[[0, 2]]
            rel_targ = rel_targ[[0, 2]]
            angle_to_target = get_angle(robot_forward, rel_targ)
            dist_to_final_nav_targ = np.linalg.norm(
                (human_pos - robot_pos)[[0, 2]]
            )

            if dist_to_final_nav_targ < self._config.dist_thresh:
                # Look at the object
                vel = compute_turn(
                    rel_pos,
                    self._config.turn_velocity * _vel_scale,
                    robot_forward,
                )
            elif angle_to_target < self._config.turn_thresh:
                # Move towards the target
                vel = [self._config.forward_velocity * _vel_scale, 0]
            else:
                # Look at the target waypoint.
                vel = compute_turn(
                    rel_targ,
                    self._config.turn_velocity * _vel_scale,
                    robot_forward,
                )

            # Update the robot's info
            base_T = vc.act(base_T, vel)
            robot_pos = np.array(base_T.translation)
            step_taken += 1

            robot_pos[1] = human_pos[1]
        return step_taken

    def _get_target_for_coord(self, obj_pos):
        start_pos = obj_pos
        if self.motion_type == "human_joints":
            self.humanoid_controller.reset(
                self.cur_articulated_agent.base_transformation
            )
        return (np.array(start_pos), np.array(obj_pos))

    def step(self, *args, **kwargs):
        max_tries = 10
        self.skill_done = False

        if self.coord_nav is None:
            self.coord_nav = self._sim.pathfinder.get_random_navigable_point(
                max_tries,
                island_index=self._sim.largest_island_idx,
            )

        kwargs[
            self._action_arg_prefix + "oracle_nav_coord_action"
        ] = self.coord_nav

        ret_val = super().step(*args, **kwargs)
        if self.skill_done:
            self.coord_nav = None

        # If the robot is nearby, the human starts to walk, otherwise, the human
        # just stops there and waits for robot to find it
        if self._config.human_stop_and_walk_to_robot_distance_threshold != -1:
            assert (
                len(self._sim.agents_mgr) == 2
            ), "Does not support more than two agents when you want human to stop and walk based on the distance to the robot"
            robot_id = int(1 - self._agent_index)
            robot_pos = self._sim.get_agent_data(
                robot_id
            ).articulated_agent.base_pos
            human_pos = self.cur_articulated_agent.base_pos
            dis = self._sim.geodesic_distance(robot_pos, human_pos)
            # The human needs to stop and wait for robot to come if the distance is too larget
            if (
                dis
                > self._config.human_stop_and_walk_to_robot_distance_threshold
            ):
                self.humanoid_controller.set_framerate_for_linspeed(
                    0.0, 0.0, self._sim.ctrl_freq
                )
            # The human needs to walk otherwise
            else:
                speed = np.random.uniform(
                    self._config.lin_speed / 5.0, self._config.lin_speed
                )
                lin_speed = speed
                ang_speed = speed
                self.humanoid_controller.set_framerate_for_linspeed(
                    lin_speed, ang_speed, self._sim.ctrl_freq
                )

        try:
            kwargs["task"].measurements.measures[
                "social_nav_stats"
            ].update_human_pos = self.coord_nav
        except Exception:
            pass
        return ret_val
