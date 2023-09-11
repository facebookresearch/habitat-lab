# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import magnum as mn
import numpy as np
from gym import spaces

import habitat_sim
from habitat.articulated_agent_controllers import HumanoidRearrangeController
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.rearrange.actions.actions import (
    BaseVelAction,
    BaseVelNonCylinderAction,
    HumanoidJointAction,
)
from habitat.tasks.rearrange.utils import place_agent_at_dist_from_pos
from habitat.tasks.utils import get_angle
from habitat_sim.physics import VelocityControl


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

        elif self.motion_type == "human_joints":
            HumanoidJointAction.__init__(self, *args, **kwargs)
            self.humanoid_controller = self.lazy_inst_humanoid_controller(
                task, config
            )

        else:
            raise ValueError("Unrecognized motion type for oracle nav  action")

        self._task = task
        self._poss_entities = (
            self._task.pddl_problem.get_ordered_entities_list()
        )
        self._prev_ep_id = None
        self._targets = {}
        self.skill_done = False
        self._spawn_max_dist_to_obj = self._config.spawn_max_dist_to_obj
        self._num_spawn_attempts = self._config.num_spawn_attempts
        self._dist_thresh = self._config.dist_thresh
        self._turn_thresh = self._config.turn_thresh
        self._turn_velocity = self._config.turn_velocity
        self._forward_velocity = self._config.forward_velocity

    @staticmethod
    def _compute_turn(rel, turn_vel, robot_forward):
        is_left = np.cross(robot_forward, rel) > 0
        if is_left:
            vel = [0, -turn_vel]
        else:
            vel = [0, turn_vel]
        return vel

    def lazy_inst_humanoid_controller(self, task, config):
        # Lazy instantiation of humanoid controller
        # We assign the task with the humanoid controller, so that multiple actions can
        # use it.

        if (
            not hasattr(task, "humanoid_controller")
            or task.humanoid_controller is None
        ):
            # Initialize humanoid controller
            agent_name = self._sim.habitat_config.agents_order[
                self._agent_index
            ]
            walk_pose_path = self._sim.habitat_config.agents[
                agent_name
            ].motion_data_path

            humanoid_controller = HumanoidRearrangeController(walk_pose_path)
            humanoid_controller.set_framerate_for_linspeed(
                config["lin_speed"], config["ang_speed"], self._sim.ctrl_freq
            )
            task.humanoid_controller = humanoid_controller
        return task.humanoid_controller

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
        self.skill_done = False

    def _get_target_for_idx(self, nav_to_target_idx: int):
        nav_to_obj = self._poss_entities[nav_to_target_idx]
        if (
            nav_to_target_idx not in self._targets
            or "robot" in nav_to_obj.name
        ):
            obj_pos = self._task.pddl_problem.sim_info.get_entity_pos(
                nav_to_obj
            )
            if "robot" in nav_to_obj.name:
                # Safety margin between the human and the robot
                sample_distance = 1.0
            else:
                sample_distance = self._spawn_max_dist_to_obj
            start_pos, _, _ = place_agent_at_dist_from_pos(
                np.array(obj_pos),
                0.0,
                sample_distance,
                self._sim,
                self._num_spawn_attempts,
                1,
                self.cur_articulated_agent,
            )

            if self.motion_type == "human_joints":
                self.humanoid_controller.reset(
                    self.cur_articulated_agent.base_transformation
                )
            self._targets[nav_to_target_idx] = (start_pos, np.array(obj_pos))
        return self._targets[nav_to_target_idx]

    def _path_to_point(self, point):
        """
        Obtain path to reach the coordinate point. If agent_pos is not given
        the path starts at the agent base pos, otherwise it starts at the agent_pos
        value
        :param point: Vector3 indicating the target point
        """
        agent_pos = self.cur_articulated_agent.base_pos

        path = habitat_sim.ShortestPath()
        path.requested_start = agent_pos
        path.requested_end = point
        found_path = self._sim.pathfinder.find_path(path)
        if not found_path:
            return [agent_pos, point]
        return path.points

    def _update_controller_to_navmesh(self):
        base_offset = self.cur_articulated_agent.params.base_offset
        prev_query_pos = self.cur_articulated_agent.base_pos
        target_query_pos = (
            self.humanoid_controller.obj_transform_base.translation
            + base_offset
        )

        filtered_query_pos = self._sim.step_filter(
            prev_query_pos, target_query_pos
        )
        fixup = filtered_query_pos - target_query_pos
        self.humanoid_controller.obj_transform_base.translation += fixup

    def step(self, *args, is_last_action, **kwargs):
        self.skill_done = False
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
        base_T = self.cur_articulated_agent.base_transformation
        curr_path_points = self._path_to_point(final_nav_targ)
        robot_pos = np.array(self.cur_articulated_agent.base_pos)

        if curr_path_points is None:
            raise Exception
        else:
            # Compute distance and angle to target
            if len(curr_path_points) == 1:
                curr_path_points += curr_path_points
            cur_nav_targ = curr_path_points[1]
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
                dist_to_final_nav_targ < self._dist_thresh
                and angle_to_obj < self._turn_thresh
            ) or dist_to_final_nav_targ < self._dist_thresh / 10.0

            if self.motion_type == "base_velocity":
                if not at_goal:
                    if dist_to_final_nav_targ < self._dist_thresh:
                        # Look at the object
                        vel = OracleNavAction._compute_turn(
                            rel_pos, self._turn_velocity, robot_forward
                        )
                    elif angle_to_target < self._turn_thresh:
                        # Move towards the target
                        vel = [self._forward_velocity, 0]
                    else:
                        # Look at the target waypoint.
                        vel = OracleNavAction._compute_turn(
                            rel_targ, self._turn_velocity, robot_forward
                        )
                else:
                    vel = [0, 0]
                    self.skill_done = True
                kwargs[f"{self._action_arg_prefix}base_vel"] = np.array(vel)
                return BaseVelAction.step(
                    self, *args, is_last_action=is_last_action, **kwargs
                )

            elif self.motion_type == "human_joints":
                # Update the humanoid base
                self.humanoid_controller.obj_transform_base = base_T
                if not at_goal:
                    if dist_to_final_nav_targ < self._dist_thresh:
                        # Look at the object
                        self.humanoid_controller.calculate_turn_pose(
                            mn.Vector3([rel_pos[0], 0.0, rel_pos[1]])
                        )
                    else:
                        # Move towards the target
                        self.humanoid_controller.calculate_walk_pose(
                            mn.Vector3([rel_targ[0], 0.0, rel_targ[1]])
                        )
                else:
                    self.humanoid_controller.calculate_stop_pose()
                    self.skill_done = True

                self._update_controller_to_navmesh()
                base_action = self.humanoid_controller.get_pose()
                kwargs[
                    f"{self._action_arg_prefix}human_joints_trans"
                ] = base_action

                return HumanoidJointAction.step(
                    self, *args, is_last_action=is_last_action, **kwargs
                )
            else:
                raise ValueError(
                    "Unrecognized motion type for oracle nav action"
                )


class SimpleVelocityControlEnv:
    """
    Simple velocity control environment for moving agent
    """

    def __init__(self, sim_freq=120.0):
        # the velocity control
        self.vel_control = VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.ang_vel_is_local = True
        self._sim_freq = sim_freq

    def act(self, trans, vel, sim=None):
        linear_velocity = vel[0]
        angular_velocity = vel[1]
        # Map velocity actions
        self.vel_control.linear_velocity = mn.Vector3(
            [linear_velocity, 0.0, 0.0]
        )
        self.vel_control.angular_velocity = mn.Vector3(
            [0.0, angular_velocity, 0.0]
        )
        # Compute the rigid state
        rigid_state = habitat_sim.RigidState(
            mn.Quaternion.from_matrix(trans.rotation()), trans.translation
        )

        # Get the target rigit state based on the simulation frequency
        target_rigid_state = self.vel_control.integrate_transform(
            1 / self._sim_freq, rigid_state
        )
        # Get the ending pos of the agent
        if sim is not None:
            end_pos = sim.step_filter(
                rigid_state.translation, target_rigid_state.translation
            )
        else:
            end_pos = target_rigid_state.translation

        # Construct the target trans
        target_trans = mn.Matrix4.from_(
            target_rigid_state.rotation.to_matrix(),
            end_pos,
        )

        return target_trans


@registry.register_task_action
class OracleNavWithBackingUpAction(BaseVelNonCylinderAction, OracleNavAction):  # type: ignore
    """
    Oracle nav action with backing-up. This function allows the robot to move
    backward to avoid obstacles.
    """

    def __init__(self, *args, task, **kwargs):
        OracleNavAction.__init__(self, *args, task=task, **kwargs)
        if self.motion_type == "base_velocity":
            BaseVelNonCylinderAction.__init__(self, *args, **kwargs, task=task)

        # Define the navigation target
        self.at_goal = False
        self.skill_done = False
        self._navmesh_offset_for_agent_placement = (
            self._config.navmesh_offset_for_agent_placement
        )
        self._navmesh_offset = self._config.navmesh_offset

        self._nav_pos_3d = [
            np.array([xz[0], 0.0, xz[1]]) for xz in self._navmesh_offset
        ]

        # Initialize the velocity controller
        self._vc = SimpleVelocityControlEnv(self._config.sim_freq)

    @property
    def action_space(self):
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "oracle_nav_with_backing_up_action": spaces.Box(
                    shape=(1,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                )
            }
        )

    def _get_target_for_idx(self, nav_to_target_idx: int):
        if nav_to_target_idx not in self._targets:
            nav_to_obj = self._poss_entities[nav_to_target_idx]
            obj_pos = self._task.pddl_problem.sim_info.get_entity_pos(
                nav_to_obj
            )
            start_pos, _, _ = place_agent_at_dist_from_pos(
                np.array(obj_pos),
                0.0,
                self._spawn_max_dist_to_obj,
                self._sim,
                self._num_spawn_attempts,
                1,
                self.cur_articulated_agent,
                self._navmesh_offset_for_agent_placement,
            )

            if self.motion_type == "human_joints":
                self.humanoid_controller.reset(
                    self.cur_articulated_agent.base_transformation
                )
            self._targets[nav_to_target_idx] = (start_pos, np.array(obj_pos))
        return self._targets[nav_to_target_idx]

    def is_collision(self, trans) -> bool:
        """
        The function checks if the agent collides with the object
        given the navmesh
        """
        cur_pos = [trans.transform_point(xyz) for xyz in self._nav_pos_3d]
        cur_pos = [
            np.array([xz[0], self.cur_articulated_agent.base_pos[1], xz[2]])
            for xz in cur_pos
        ]

        for pos in cur_pos:  # noqa: SIM110
            # Return true if the pathfinder says it is not navigable
            if not self._sim.pathfinder.is_navigable(pos):
                return True

        return False

    def rotation_collision_check(
        self,
        next_pos,
    ):
        """
        This function checks if the robot needs to do backing-up action
        """
        # Make a copy of agent trans
        trans = mn.Matrix4(self.cur_articulated_agent.sim_obj.transformation)
        angle = float("inf")
        # Get the current location of the agent
        cur_pos = self.cur_articulated_agent.base_pos
        # Set the trans to be agent location
        trans.translation = self.cur_articulated_agent.base_pos

        while abs(angle) > self._turn_thresh:
            # Compute the robot facing orientation
            rel_pos = (next_pos - cur_pos)[[0, 2]]
            forward = np.array([1.0, 0, 0])
            robot_forward = np.array(trans.transform_vector(forward))
            robot_forward = robot_forward[[0, 2]]
            angle = get_angle(robot_forward, rel_pos)
            vel = OracleNavAction._compute_turn(
                rel_pos, self._turn_velocity, robot_forward
            )
            trans = self._vc.act(trans, vel)
            cur_pos = trans.translation

            if self.is_collision(trans):
                return True

        return False

    def step(self, *args, is_last_action, **kwargs):
        self.skill_done = False
        nav_to_target_idx = kwargs[
            self._action_arg_prefix + "oracle_nav_with_backing_up_action"
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
        # Get the base transformation
        base_T = self.cur_articulated_agent.base_transformation
        # Get the current path
        curr_path_points = self._path_to_point(final_nav_targ)
        # Get the robot position
        robot_pos = np.array(self.cur_articulated_agent.base_pos)

        # Get the current robot/human pos assuming human is agent 1
        robot_human_dis = None

        if self._sim.num_articulated_agents > 1:
            # This is very specific to SIRo. Careful merging
            _robot_pos = np.array(
                self._sim.get_agent_data(
                    0
                ).articulated_agent.base_transformation.translation
            )[[0, 2]]
            _human_pos = np.array(
                self._sim.get_agent_data(
                    1
                ).articulated_agent.base_transformation.translation
            )[[0, 2]]
            # Compute the distance
            robot_human_dis = np.linalg.norm(_robot_pos - _human_pos)

        if curr_path_points is None:
            raise RuntimeError("Pathfinder returns empty list")
        else:
            # Compute distance and angle to target
            if len(curr_path_points) == 1:
                curr_path_points += curr_path_points

            cur_nav_targ = curr_path_points[1]
            forward = np.array([1.0, 0, 0])
            robot_forward = np.array(base_T.transform_vector(forward))

            # Compute relative target
            rel_targ = cur_nav_targ - robot_pos

            # Compute heading angle (2D calculation)
            robot_forward = robot_forward[[0, 2]]
            rel_targ = rel_targ[[0, 2]]
            rel_pos = (obj_targ_pos - robot_pos)[[0, 2]]
            # Get the angles
            angle_to_target = get_angle(robot_forward, rel_targ)
            angle_to_obj = get_angle(robot_forward, rel_pos)
            # Compute the distance
            dist_to_final_nav_targ = np.linalg.norm(
                (final_nav_targ - robot_pos)[[0, 2]]
            )
            at_goal = (
                dist_to_final_nav_targ < self._dist_thresh
                and angle_to_obj < self._turn_thresh
            ) or dist_to_final_nav_targ < self._dist_thresh / 10.0

            # Planning to see if the robot needs to do back-up
            need_move_backward = False
            # if (
            #     dist_to_final_nav_targ >= self._dist_thresh
            #     and angle_to_target >= self._turn_thresh
            #     and not at_goal
            # ):
            #     # check if there is a collision caused by rotation
            #     # if it does, we should block the rotation, and
            #     # only move backward
            #     need_move_backward = self.rotation_collision_check(
            #         cur_nav_targ,
            #     )

            if need_move_backward:
                # Backward direction
                forward = np.array([-1.0, 0, 0])
                robot_forward = np.array(base_T.transform_vector(forward))
                # Compute relative target
                rel_targ = cur_nav_targ - robot_pos
                # Compute heading angle (2D calculation)
                robot_forward = robot_forward[[0, 2]]
                rel_targ = rel_targ[[0, 2]]
                rel_pos = (obj_targ_pos - robot_pos)[[0, 2]]
                # Get the angles
                angle_to_target = get_angle(robot_forward, rel_targ)
                angle_to_obj = get_angle(robot_forward, rel_pos)
                # Compute the distance
                dist_to_final_nav_targ = np.linalg.norm(
                    (final_nav_targ - robot_pos)[[0, 2]]
                )
                at_goal = (
                    dist_to_final_nav_targ < self._dist_thresh
                    and angle_to_obj < self._turn_thresh
                )

            if self.motion_type == "base_velocity":
                if not at_goal:
                    self.at_goal = False
                    if self.nav_mode == "avoid":
                        backward = np.array([-1.0, 0, 0])
                        robot_backward = np.array(
                            base_T.transform_vector(backward)
                        )
                        robot_backward = robot_backward[[0, 2]]
                        angle_to_target = get_angle(robot_backward, rel_targ)
                        if (
                            self.simple_backward
                            or angle_to_target < self._turn_thresh
                        ):
                            # Move backwards the target
                            vel = [-self._forward_velocity, 0]
                        else:
                            # Robot's rear looks at the target waypoint.
                            vel = OracleNavAction._compute_turn(
                                rel_targ, self._turn_velocity, robot_backward
                            )
                    else:
                        if dist_to_final_nav_targ < self._dist_thresh:
                            # Look at the object
                            vel = OracleNavAction._compute_turn(
                                rel_pos, self._turn_velocity, robot_forward
                            )
                        elif angle_to_target < self._turn_thresh:
                            # Move towards the target
                            vel = [self._forward_velocity, 0]
                        else:
                            # Look at the target waypoint.
                            vel = OracleNavAction._compute_turn(
                                rel_targ, self._turn_velocity, robot_forward
                            )
                else:
                    self.at_goal = True
                    self.skill_done = True
                    vel = [0, 0]

                if need_move_backward:
                    vel[0] = -1 * vel[0]

                # If the human and robot are too close to each other, pause the robot
                if (
                    self._config.agents_dist_thresh != -1
                    and robot_human_dis is not None
                    and robot_human_dis < self._config.agents_dist_thresh
                ):
                    vel = [0, 0]

                kwargs[f"{self._action_arg_prefix}base_vel"] = np.array(vel)
                return BaseVelNonCylinderAction.step(
                    self, *args, is_last_action=is_last_action, **kwargs
                )

            elif self.motion_type == "human_joints":
                # Update the humanoid base
                self.humanoid_controller.obj_transform_base = base_T
                if not at_goal:
                    self.at_goal = False
                    if dist_to_final_nav_targ < self._dist_thresh:
                        # Look at the object
                        self.humanoid_controller.calculate_turn_pose(
                            mn.Vector3([rel_pos[0], 0.0, rel_pos[1]])
                        )
                    else:
                        # Move towards the target
                        self.humanoid_controller.calculate_walk_pose(
                            mn.Vector3([rel_targ[0], 0.0, rel_targ[1]])
                        )
                else:
                    self.at_goal = True
                    self.skill_done = True
                    self.humanoid_controller.calculate_stop_pose()

                self._update_controller_to_navmesh()
                base_action = self.humanoid_controller.get_pose()
                kwargs[
                    f"{self._action_arg_prefix}human_joints_trans"
                ] = base_action

                return HumanoidJointAction.step(
                    self, *args, is_last_action=is_last_action, **kwargs
                )
            else:
                raise ValueError(
                    "Unrecognized motion type for oracle nav action"
                )


@registry.register_task_action
class OracleNavCoordAction(OracleNavAction):
    """
    An action that will convert the index of an entity (in the sense of
    `PddlEntity`) to navigate to and convert this to base/humanoid joint control to move the
    robot to the closest navigable position to that entity. The entity index is
    the index into the list of all available entities in the current scene. The
    config flag motion_type indicates whether the low level action will be a base_velocity or
    a joint control.
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
        precision = 0.25
        pos_hash = np.around(obj_pos / precision, decimals=0) * precision
        pos_hash = tuple(pos_hash)
        if pos_hash not in self._targets:
            start_pos, _, _ = place_agent_at_dist_from_pos(
                np.array(obj_pos),
                0.0,
                self._spawn_max_dist_to_obj,
                self._sim,
                self._num_spawn_attempts,
                1,
                self.cur_articulated_agent,
            )
            self._targets[pos_hash] = start_pos
        else:
            start_pos = self._targets[pos_hash]
        if self.motion_type == "human_joints":
            self.humanoid_controller.reset(
                self.cur_articulated_agent.base_transformation
            )
        return (start_pos, np.array(obj_pos))

    def step(self, *args, is_last_action, **kwargs):
        self.skill_done = False
        # TODO better way to handle this
        try:
            nav_to_target_coord = kwargs[
                self._action_arg_prefix + "oracle_nav_coord_action"
            ]
        except Exception:
            nav_to_target_coord = kwargs[
                self._action_arg_prefix + "oracle_nav_human_action"
            ]
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
            cur_nav_targ = curr_path_points[1]
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
                dist_to_final_nav_targ < self._dist_thresh
                and angle_to_obj < self._turn_thresh
            ) or dist_to_final_nav_targ < self._dist_thresh / 10.0
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
                            or angle_to_target < self._turn_thresh
                        ):
                            # Move backwards the target
                            vel = [-self._forward_velocity, 0]
                        else:
                            # Robot's rear looks at the target waypoint.
                            vel = OracleNavAction._compute_turn(
                                rel_targ, self._turn_velocity, robot_backward
                            )
                    else:
                        if dist_to_final_nav_targ < self._dist_thresh:
                            # Look at the object
                            vel = OracleNavAction._compute_turn(
                                rel_pos, self._turn_velocity, robot_forward
                            )
                        elif angle_to_target < self._turn_thresh:
                            # Move towards the target
                            vel = [self._forward_velocity, 0]
                        else:
                            # Look at the target waypoint.
                            vel = OracleNavAction._compute_turn(
                                rel_targ, self._turn_velocity, robot_forward
                            )
                else:
                    vel = [0, 0]
                    self.skill_done = True
                kwargs[f"{self._action_arg_prefix}base_vel"] = np.array(vel)
                return BaseVelAction.step(
                    self, *args, is_last_action=is_last_action, **kwargs
                )

            elif self.motion_type == "human_joints":
                # Update the humanoid base
                self.humanoid_controller.obj_transform_base = base_T
                if not at_goal:
                    if dist_to_final_nav_targ < self._dist_thresh:
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
                self._update_controller_to_navmesh()
                base_action = self.humanoid_controller.get_pose()
                kwargs[
                    f"{self._action_arg_prefix}human_joints_trans"
                ] = base_action

                return HumanoidJointAction.step(
                    self, *args, is_last_action=is_last_action, **kwargs
                )
            else:
                raise ValueError(
                    "Unrecognized motion type for oracle nav action"
                )


@registry.register_task_action
class OracleNavCoordActionNonCylinder(OracleNavWithBackingUpAction):
    """
    An action that will convert the index of an entity (in the sense of
    `PddlEntity`) to navigate to and convert this to base/humanoid joint control to move the
    robot to the closest navigable position to that entity. The entity index is
    the index into the list of all available entities in the current scene. The
    config flag motion_type indicates whether the low level action will be a base_velocity or
    a joint control.
    """

    def __init__(self, *args, task, **kwargs):
        super().__init__(*args, task=task, **kwargs)
        self.nav_mode = None
        self.simple_backward = False
        self.old_human_pos = None

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
        precision = 0.25
        pos_hash = np.around(obj_pos / precision, decimals=0) * precision
        pos_hash = tuple(pos_hash)
        if pos_hash not in self._targets:
            start_pos, _, _ = place_agent_at_dist_from_pos(
                np.array(obj_pos),
                0.0,
                self._spawn_max_dist_to_obj,
                self._sim,
                self._num_spawn_attempts,
                1,
                self.cur_articulated_agent,
            )
            self._targets[pos_hash] = start_pos
        else:
            start_pos = self._targets[pos_hash]
        if self.motion_type == "human_joints":
            self.humanoid_controller.reset(
                self.cur_articulated_agent.base_transformation
            )
        return (start_pos, np.array(obj_pos))

    def update_rel_targ_obstacle(
        self, rel_targ, new_human_pos, old_human_pos=None
    ):
        if old_human_pos is None:
            human_velocity_scale = 0.0
        else:
            # take the norm of the distance between old and new human position
            human_velocity_scale = float(
                np.linalg.norm(new_human_pos - old_human_pos) / 0.25
            )  # 0.25 is a magic number
            # set a minimum value for the human velocity scale
            human_velocity_scale = max(human_velocity_scale, 0.5)

        std = 8.0
        # scale the amplitude by the human velocity
        amp = 8.0 * human_velocity_scale

        # Get the position of the other agents
        other_agent_rel_pos, other_agent_dist = [], []
        curr_agent_T = np.array(
            self.cur_articulated_agent.base_transformation.translation
        )[[0, 2]]

        other_agent_rel_pos.append(rel_targ[None, :])
        other_agent_dist.append(0.0)  # dummy value
        rel_pos = new_human_pos - curr_agent_T
        dist_pos = np.linalg.norm(rel_pos)
        # normalized relative vector
        rel_pos = rel_pos / dist_pos
        other_agent_dist.append(float(dist_pos))
        other_agent_rel_pos.append(-rel_pos[None, :])

        rel_pos = np.concatenate(other_agent_rel_pos)
        rel_dist = np.array(other_agent_dist)
        weight = amp * np.exp(-(rel_dist**2) / std)
        weight[0] = 1.0
        # TODO: explore softmax?
        weight_norm = weight[:, None] / weight.sum()
        # weighted sum of the old target position and
        # relative position that avoids human
        final_rel_pos = (rel_pos * weight_norm).sum(0)
        return final_rel_pos

    def step(self, *args, is_last_action, **kwargs):
        self.skill_done = False
        # TODO better way to handle this
        try:
            nav_to_target_coord = kwargs[
                self._action_arg_prefix + "oracle_nav_coord_action"
            ]
        except Exception:
            nav_to_target_coord = kwargs[
                self._action_arg_prefix + "oracle_nav_human_action"
            ]
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
            cur_nav_targ = curr_path_points[1]
            forward = np.array([1.0, 0, 0])
            robot_forward = np.array(base_T.transform_vector(forward))

            # Compute relative target.
            rel_targ = cur_nav_targ - robot_pos

            # Compute heading angle (2D calculation)
            robot_forward = robot_forward[[0, 2]]
            rel_targ = rel_targ[[0, 2]]
            rel_pos = (obj_targ_pos - robot_pos)[[0, 2]]

            ###############
            # NEW: We will update the rel_targ position to avoid the humanoid
            # rel_targ is the next position that the agent wants to walk to
            old_rel_targ = rel_targ
            if self._sim.num_articulated_agents > 1:
                # This is very specific to SIRo. Careful merging
                for agent_index in range(self._sim.num_articulated_agents):
                    if self._agent_index == agent_index:
                        continue
                    new_human_pos = np.array(
                        self._sim.get_agent_data(
                            agent_index
                        ).articulated_agent.base_transformation.translation
                    )[[0, 2]]

            rel_targ = self.update_rel_targ_obstacle(
                rel_targ, new_human_pos, self.old_human_pos
            )
            self.old_human_pos = new_human_pos

            # NEW: If avoiding the human makes us change dir, we will
            # go backwards at times to avoid rotating
            dot_prod_rel_targ = (rel_targ * old_rel_targ).sum()
            did_change_dir = dot_prod_rel_targ < 0
            ###############

            angle_to_target = get_angle(robot_forward, rel_targ)
            angle_to_obj = get_angle(robot_forward, rel_pos)

            dist_to_final_nav_targ = np.linalg.norm(
                (final_nav_targ - robot_pos)[[0, 2]]
            )
            at_goal = (
                dist_to_final_nav_targ < self._dist_thresh
                and angle_to_obj < self._turn_thresh
            ) or dist_to_final_nav_targ < self._dist_thresh / 10.0

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
                            or angle_to_target < self._turn_thresh
                        ):
                            # Move backwards the target
                            vel = [-self._forward_velocity, 0]
                        else:
                            # Robot's rear looks at the target waypoint.
                            vel = OracleNavAction._compute_turn(
                                rel_targ, self._turn_velocity, robot_backward
                            )
                    elif self.nav_mode == "seek":
                        if dist_to_final_nav_targ < self._dist_thresh:
                            # Look at the object
                            vel = OracleNavAction._compute_turn(
                                rel_pos, self._turn_velocity, robot_forward
                            )
                        elif angle_to_target < self._turn_thresh:
                            # Move towards the target
                            vel = [self._forward_velocity, 0]
                        else:
                            # Look at the target waypoint.
                            vel = OracleNavAction._compute_turn(
                                rel_targ, self._turn_velocity, robot_forward
                            )
                    else:
                        vel = [0, 0]

                    # if dist_to_final_nav_targ < self._dist_thresh:
                    #     # Look at the object
                    #     vel = OracleNavAction._compute_turn(
                    #         rel_pos, self._turn_velocity, robot_forward
                    #     )
                    # elif angle_to_target < self._turn_thresh:
                    #     # Move towards the target
                    #     vel = [self._forward_velocity, 0]
                    # else:
                    #     # Look at the target waypoint.
                    #     if did_change_dir:
                    #         if (np.pi - angle_to_target) < self._turn_thresh:
                    #             # Move towards the target
                    #             vel = [-self._forward_velocity, 0]
                    #         else:
                    #             vel = OracleNavAction._compute_turn(
                    #                 -rel_targ,
                    #                 self._turn_velocity,
                    #                 robot_forward,
                    #             )
                    #     else:
                    #         vel = OracleNavAction._compute_turn(
                    #             rel_targ, self._turn_velocity, robot_forward
                    #         )
                else:
                    vel = [0, 0]
                    self.skill_done = True
                kwargs[f"{self._action_arg_prefix}base_vel"] = np.array(vel)
                return BaseVelNonCylinderAction.step(
                    self, *args, is_last_action=is_last_action, **kwargs
                )

            elif self.motion_type == "human_joints":
                # Update the humanoid base
                self.humanoid_controller.obj_transform_base = base_T
                if not at_goal:
                    if dist_to_final_nav_targ < self._dist_thresh:
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
                self._update_controller_to_navmesh()
                base_action = self.humanoid_controller.get_pose()
                kwargs[
                    f"{self._action_arg_prefix}human_joints_trans"
                ] = base_action

                return HumanoidJointAction.step(
                    self, *args, is_last_action=is_last_action, **kwargs
                )
            else:
                raise ValueError(
                    "Unrecognized motion type for oracle nav action"
                )


@registry.register_task_action
class OracleNavRandCoordAction(OracleNavCoordAction):
    """
    Oracle Nav RandCoord Action. Selects a random position in the scene and navigates
    there until reaching. When the arg is 1, does replanning.
    """

    def __init__(self, *args, task, **kwargs):
        super().__init__(*args, task=task, **kwargs)
        self.random_seed_counter = None

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
        self.random_seed_counter = int(self._task._episode_id)

    def _find_path_given_start_end(self, start, end):
        path = habitat_sim.ShortestPath()
        path.requested_start = start
        path.requested_end = end
        found_path = self._sim.pathfinder.find_path(path)
        if not found_path:
            return [start, end]
        return path.points

    def _reach_human(self, robot_pos, human_pos, base_T):
        # For computing facing to human
        vector_human_robot = human_pos[[0, 2]] - robot_pos[[0, 2]]
        vector_human_robot = vector_human_robot / np.linalg.norm(
            vector_human_robot
        )
        forward_robot = np.array(base_T.transform_vector(mn.Vector3(1, 0, 0)))[
            [0, 2]
        ]
        forward_robot = forward_robot / np.linalg.norm(forward_robot)
        facing = np.dot(forward_robot, vector_human_robot) > 0.5

        # Use geodesic distance here
        dis = self._sim.geodesic_distance(robot_pos, human_pos)

        # np.linalg.norm((robot_pos - human_pos)[[0, 2]])
        return dis <= 2.0 and facing

    def _compute_robot_to_human_min_step(
        self, robot_trans, human_pos, human_pos_list
    ):
        _vel_scale = 10.0

        # Copy the robot transformation
        base_T = mn.Matrix4(robot_trans)

        vc = SimpleVelocityControlEnv()

        # human_pos = human_pos_list[0]

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

            if dist_to_final_nav_targ < self._dist_thresh:
                # Look at the object
                vel = OracleNavAction._compute_turn(
                    rel_pos, self._turn_velocity * _vel_scale, robot_forward
                )
            elif angle_to_target < self._turn_thresh:
                # Move towards the target
                vel = [self._forward_velocity * _vel_scale, 0]
            else:
                # Look at the target waypoint.
                vel = OracleNavAction._compute_turn(
                    rel_targ, self._turn_velocity * _vel_scale, robot_forward
                )

            # Update the robot's info
            base_T = vc.act(base_T, vel, self._sim)
            robot_pos = np.array(base_T.translation)
            step_taken += 1

            # human_pos = human_pos_list[step_taken] if step_taken < len(human_pos_list) else human_pos_list[-1]
            robot_pos[1] = human_pos[1]
        return step_taken

    def _get_target_for_coord(self, obj_pos):
        start_pos = obj_pos
        return (start_pos, np.array(obj_pos))

    def step(self, *args, is_last_action, **kwargs):
        max_tries = 10
        self.skill_done = False

        if self.coord_nav is None:
            self._sim.seed(self.random_seed_counter)
            self.coord_nav = self._sim.pathfinder.get_random_navigable_point(
                max_tries,
                island_index=self._sim._largest_island_idx,
            )
            self.random_seed_counter += 1
        kwargs[
            self._action_arg_prefix + "oracle_nav_coord_action"
        ] = self.coord_nav
        kwargs["is_last_action"] = is_last_action
        ret_val = super().step(*args, is_last_action, **kwargs)
        if self.skill_done:
            self.coord_nav = None

        try:
            kwargs["task"].measurements.measures[
                "social_nav_stats"
            ].update_human_pos = self.coord_nav
        except Exception:
            pass
        return ret_val


@registry.register_task_action
# class OracleNavHumanAction(OracleNavCoordActionNonCylinder):
class OracleNavHumanAction(OracleNavCoordAction):
    """
    Oracle Nav human Action. Selects a random position in the scene and navigates
    there until reaching. When the arg is 1, does replanning.
    """

    def __init__(self, *args, task, **kwargs):
        super().__init__(*args, task=task, **kwargs)
        self.nav_mode = "seek"
        self.first_avoid = True
        self.continue_avoid_count = 0
        self.prev_pos = None
        self.target_pos = None
        self.prev_target_pos = None
        self.prev_human_goal = None
        self.config = kwargs["config"]

    @property
    def action_space(self):
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "oracle_nav_human_action": spaces.Box(
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
        self.human_pos = None
        self.first_avoid = True
        self.continue_avoid_count = 0
        self.prev_pos = None
        self.target_pos = None
        self.prev_target_pos = None
        self.prev_human_goal = None

    def _get_target_for_coord(self, obj_pos):
        start_pos = obj_pos
        return (start_pos, np.array(obj_pos))

    def _check_point_overlap(
        self, human_pos, human_goal, robot_pos, robot_goal
    ):
        path_human = habitat_sim.ShortestPath()
        path_human.requested_start = human_pos
        path_human.requested_end = human_goal
        self._sim.pathfinder.find_path(path_human)

        path_robot = habitat_sim.ShortestPath()
        path_robot.requested_start = robot_pos
        path_robot.requested_end = robot_goal
        self._sim.pathfinder.find_path(path_robot)

        dist_interval = 0.1

        _path_robot = []
        for i in range(len(path_robot.points) - 1):
            start_pt = path_robot.points[i][[0, 2]]
            end_pt = path_robot.points[i + 1][[0, 2]]
            pts = int(np.linalg.norm((start_pt - end_pt)) / dist_interval) + 2
            new_x = np.linspace(start_pt[0], end_pt[0], pts)
            new_y = np.linspace(start_pt[1], end_pt[1], pts)
            for j in range(pts):
                _path_robot.append(np.array([new_x[j], new_y[j]]))

        _path_human = []
        for i in range(len(path_human.points) - 1):
            start_pt = path_human.points[i][[0, 2]]
            end_pt = path_human.points[i + 1][[0, 2]]
            pts = int(np.linalg.norm((start_pt - end_pt)) / dist_interval) + 2
            new_x = np.linspace(start_pt[0], end_pt[0], pts)
            new_y = np.linspace(start_pt[1], end_pt[1], pts)
            for j in range(pts):
                _path_human.append(np.array([new_x[j], new_y[j]]))

        for i in range(len(_path_human)):
            if i >= len(_path_robot):
                break
            rp = _path_robot[i]
            hp = _path_human[i]
            if np.linalg.norm((rp - hp)) < 1.0:
                return True
        return False

    def step(self, *args, is_last_action, **kwargs):
        # Hyperparameter
        max_tries = 100

        dis_to_avoid_human = self.config.dis_to_avoid_human
        target_radius_near_human = self.config.target_radius_near_human
        target_radius_near_robot = self.config.target_radius_near_robot
        self.simple_backward = True

        self.skill_done = False

        # Get the position of the agents
        human_pos = np.array(
            self._sim.get_agent_data(1).articulated_agent.base_pos
        )
        robot_pos = np.array(
            self._sim.get_agent_data(0).articulated_agent.base_pos
        )

        dis = np.linalg.norm((human_pos - robot_pos)[[0, 2]])

        social_nav_stats = (
            kwargs["task"]
            .measurements.measures["social_nav_stats"]
            .get_metric()
        )
        # human_in_frame = social_nav_stats["found_human"]
        # human_roating = social_nav_stats["human_rotate"]

        human_goal_x = (
            kwargs["task"]
            .measurements.measures["social_nav_stats"]
            .get_metric()["human_goal_x"]
        )
        human_goal_y = (
            kwargs["task"]
            .measurements.measures["social_nav_stats"]
            .get_metric()["human_goal_y"]
        )
        human_goal_z = (
            kwargs["task"]
            .measurements.measures["social_nav_stats"]
            .get_metric()["human_goal_z"]
        )
        human_goal = np.array([human_goal_x, human_goal_y, human_goal_z])

        if dis > dis_to_avoid_human:
            self.target_pos = (
                self._sim.pathfinder.get_random_navigable_point_near(
                    circle_center=human_pos,
                    radius=target_radius_near_human,
                    max_tries=max_tries,
                    island_index=self._sim._largest_island_idx,
                )
            )
            self.target_pos = human_pos
            self.nav_mode = "seek"
            self.first_avoid = True
        else:  # if self.first_avoid:
            dist_to_cur_goal = np.linalg.norm(self.target_pos - robot_pos)
            goal_diff = np.linalg.norm(self.prev_human_goal - human_goal)
            if self.first_avoid or goal_diff > 0.5:
                v_robot_to_human = human_pos - robot_pos
                v_robot_to_human = v_robot_to_human / np.linalg.norm(
                    v_robot_to_human
                )
                not_found = True
                offset = 0.0
                try_times = 0
                while not_found:
                    self.target_pos = (
                        self._sim.pathfinder.get_random_navigable_point_near(
                            circle_center=robot_pos,
                            radius=target_radius_near_robot - offset,
                            max_tries=max_tries,
                            island_index=self._sim._largest_island_idx,
                        )
                    )

                    # self.target_pos = self._sim.pathfinder.get_random_navigable_point(island_index=self._sim._largest_island_idx)

                    if np.isnan(self.target_pos[0]):
                        self.target_pos = self.prev_target_pos
                        break
                    v_robot_to_target = np.array(self.target_pos) - robot_pos
                    v_robot_to_target = v_robot_to_target / np.linalg.norm(
                        v_robot_to_target
                    )
                    # Make sure the new target point has oppsite direction
                    # to human
                    if np.dot(
                        v_robot_to_human, v_robot_to_target
                    ) < 0.0 and not self._check_point_overlap(
                        human_pos, human_goal, robot_pos, self.target_pos
                    ):
                        not_found = False

                    # if not self._check_point_overlap(
                    #     human_pos, human_goal, robot_pos, self.target_pos
                    # ):
                    #     not_found = False

                    # print("loop")

                    try_times += 1
                    if try_times > 1000:
                        self.target_pos = self.prev_target_pos
                        break

            self.nav_mode = "avoid"
            self.first_avoid = False

        # Robot should stop moving if it is in the avoid mode
        # if human_roating and self.nav_mode == "avoid":
        #     self.nav_mode = "stop"
        # print("self.nav_mode:", self.nav_mode)
        # Set the position
        kwargs[
            self._action_arg_prefix + "oracle_nav_human_action"
        ] = self.target_pos
        kwargs["is_last_action"] = is_last_action
        ret_val = super().step(*args, is_last_action, **kwargs)

        if not np.isnan(self.target_pos[0]):
            self.prev_target_pos = self.target_pos

        self.prev_human_goal = human_goal

        return ret_val
