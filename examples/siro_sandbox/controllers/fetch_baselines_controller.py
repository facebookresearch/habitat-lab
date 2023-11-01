#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import TYPE_CHECKING

import magnum as mn
import numpy as np
import torch

import habitat_sim
from habitat.core.spaces import ActionSpace
from habitat.datasets.rearrange import navmesh_utils
from habitat.tasks.rearrange.utils import get_aabb
from habitat.tasks.utils import cartesian_to_polar
from habitat_baselines.rl.hrl.utils import find_action_range
from habitat_baselines.utils.common import get_num_actions
from habitat_sim.physics import CollisionGroups, MotionType

from .baselines_controller import SingleAgentBaselinesController

if TYPE_CHECKING:
    import habitat


class FetchState(Enum):
    WAIT = 1
    SEARCH = 2
    PICK = 3
    BRING = 4
    DROP = 5
    RESET_ARM_BEFORE_WAIT = 6
    SEARCH_ORACLE_NAV = 7
    BRING_ORACLE_NAV = 8
    BEG_RESET = 9
    SEARCH_TIMEOUT_WAIT = 10
    BRING_TIMEOUT_WAIT = 11
    FOLLOW = 12


class FollowStatePolicy(Enum):
    IDLE = 1
    POINT_NAV = 2
    SOCIAL_NAV = 3


# The hyper-parameters for the state machine
PICK_DIST_THRESHOLD = 1.0  # Can use 1.2
DROP_DIST_THRESHOLD = 1.0
IS_ACCESSIBLE_THRESHOLD = 1.25
ROBOT_BASE_HEIGHT = 0.53  # Default: 0.6043850
TOTAL_BEG_MOTION = 150  # This number cannot be smaller than 30
LOCAL_PLACE_TARGET = [
    -0.05,
    0.0,
    0.25,
]  # Arm local ee location format: [up,right,front]
START_FETCH_OBJ_VEL_THRESHOLD = (
    1.5  # The object velocity threshold to start to search for the object
)
START_FETCH_OBJ_DIS_THRESHOLD = (
    1.8  # if the human is too close to the object, they block Spot
)
START_FETCH_ROBOT_DIS_THRESHOLD = (
    1.8  # if the human is too close to Spot, they block Spot
)
FOLLOW_SWITCH_GEO_DIS_FOR_POINT_SOCIAL_NAV = 2.5
PICK_STEPS = 40


class FetchBaselinesController(SingleAgentBaselinesController):
    def __init__(
        self, agent_idx, is_multi_agent, config, gym_env, habitat_env
    ):
        self._current_state = FetchState.WAIT
        self.should_start_skill = False
        self.object_interest_id = None
        self.rigid_obj_interest = None
        self._target_place_trans = None
        self.grasped_object_id = None
        self.grasped_object = None
        self._last_object_drop_info = None
        self._env = gym_env
        self._thrown_object_collision_group = CollisionGroups.UserGroup7
        self.counter_pick = 0
        self._habitat_env: habitat.Env = habitat_env  # type: ignore
        self._pick_dist_threshold = PICK_DIST_THRESHOLD
        self._drop_dist_threshold = DROP_DIST_THRESHOLD
        self._can_pick_for_ray_threshold = 1.0
        self._local_place_target = LOCAL_PLACE_TARGET
        super().__init__(agent_idx, is_multi_agent, config, gym_env)
        self._policy_info = self._init_policy_input()
        self.defined_skills = self._config.habitat_baselines.rl.policy[
            self._agent_name
        ].hierarchical_policy.defined_skills
        self._use_pick_skill_as_place_skill = (
            self.defined_skills.place.use_pick_skill_as_place_skill
        )
        self._skill_steps = 0
        self._beg_state_lock = False
        self._robot_obj_T = None
        self.safe_pos = None
        self.gt_path = None
        self.human_block_robot_when_searching = False
        # Flag for controlling either point nav or social nav to use in FOLLOW state
        self._follow_state_policy_chosen = FollowStatePolicy.IDLE

    def _init_policy_input(self):
        prev_actions = torch.zeros(
            self._num_envs,
            *self._action_shape,
            device=self.device,
            dtype=torch.long if self._discrete_actions else torch.float,
        )
        rnn_hidden_states = torch.zeros(
            self._num_envs, *self._agent.hidden_state_shape
        )

        policy_info = {
            "rnn_hidden_states": rnn_hidden_states,
            "prev_actions": prev_actions,
            "masks": torch.ones(1, device=self.device, dtype=torch.bool),
            "cur_batch_idx": torch.zeros(
                1, device=self.device, dtype=torch.long
            ),
        }
        return {"ll_policy": policy_info}

    def cancel_fetch(self, skip_reset_arm=False):
        if self.grasped_object:
            env = self._habitat_env

            # Open gripper
            self._get_grasp_mgr(env).desnap()

            self.grasped_object.motion_type = MotionType.DYNAMIC

            grasped_rigid_obj = self.grasped_object

            obj_bb = get_aabb(self.grasped_object_id, env._sim)
            self._last_object_drop_info = (
                grasped_rigid_obj,
                max(obj_bb.size_x(), obj_bb.size_y(), obj_bb.size_z()),
            )
            self.grasped_object_id = None
            self.grasped_object = None

        # Make sure that we reset the arm when the robot is in such state that involves the arm movement
        if (
            self.current_state == FetchState.PICK
            or self.current_state == FetchState.DROP
        ) and not skip_reset_arm:
            self.current_state = FetchState.RESET_ARM_BEFORE_WAIT
        else:
            self.current_state = FetchState.WAIT
        self.counter_pick = 0
        self.object_interest_id = None
        self.rigid_obj_interest = None
        self._target_place_trans = None
        # For the safe snap point visualization purpose
        self.safe_pos = None
        self.gt_path = None

    # todo: make this non-public, since user code shouldn't be able to set arbitrary states
    @property
    def current_state(self):
        return self._current_state

    @current_state.setter
    def current_state(self, value):
        if self._current_state != value:
            self.should_start_skill = True
        else:
            self.should_start_skill = False
        self._current_state = value

    def _get_grasp_mgr(self, env):
        agents_mgr = env._sim.agents_mgr
        grasp_mgr = agents_mgr._all_agent_data[self._agent_idx].grasp_mgr
        return grasp_mgr

    def get_articulated_agent(self):
        return self._habitat_env._sim.agents_mgr[  # type: ignore
            self._agent_idx
        ].articulated_agent

    def start_skill(self, observations, skill_name):
        skill_walk = self._agent.actor_critic._skills[  # type: ignore
            self._agent.actor_critic._name_to_idx[skill_name]  # type: ignore
        ]
        policy_input = self._policy_info["ll_policy"]
        obs = self._batch_and_apply_transforms([observations])
        prev_actions = policy_input["prev_actions"]
        rnn_hidden_states = policy_input["rnn_hidden_states"]
        batch_id = 0
        if skill_name == "pick":
            skill_args = ["0|0"]
        elif skill_name == "place":
            skill_args = ["0|0", "0|0"]
        else:
            skill_args = ["", "0|0", ""]
        (
            rnn_hidden_states[batch_id],
            prev_actions[batch_id],
        ) = skill_walk.on_enter(
            [skill_args], [batch_id], obs, rnn_hidden_states, prev_actions
        )
        self.should_start_skill = False
        self._skill_steps = 0

    def get_cartesian_obj_coords(self, obj_trans):
        articulated_agent_T = self.get_articulated_agent().base_transformation
        rel_pos = articulated_agent_T.inverted().transform_point(obj_trans)
        rho, phi = cartesian_to_polar(rel_pos[0], rel_pos[1])

        return rho, phi

    def get_geodesic_distance_obj_coords(self, obj_trans):
        # Make sure the point is a navigatable waypoint
        obj_trans = self._habitat_env._sim.safe_snap_point(obj_trans)  # type: ignore
        _, phi = self.get_cartesian_obj_coords(obj_trans)
        rho = self._habitat_env._sim.geodesic_distance(
            obj_trans,
            self.get_articulated_agent().base_transformation.translation,
        )

        return rho, phi

    def check_if_skill_done(self, observations, skill_name):
        """Check if the skill is done"""
        ll_skil = self._agent.actor_critic._skills[  # type: ignore
            self._agent.actor_critic._name_to_idx[skill_name]  # type: ignore
        ]
        return ll_skil._is_skill_done(
            self._batch_and_apply_transforms([observations])
        )

    def get_place_loc(self, env, target):
        global_T = env._sim.get_agent_data(
            self._agent_idx
        ).articulated_agent.ee_transform()

        return np.array(global_T.transform_point(np.array(target)))

    def get_ee_loc(self, env):
        global_T = env._sim.get_agent_data(
            self._agent_idx
        ).articulated_agent.ee_transform()
        return global_T.translation

    def _check_obj_ray_to_ee(self, obj_trans, env):
        """Cast a ray from the object to the robot"""
        ee_pos = (
            env._sim.get_agent_data(self._agent_idx)
            .articulated_agent.ee_transform()
            .translation
        )
        obj_to_ee_vec = ee_pos - obj_trans
        ray_result = env._sim.cast_ray(
            habitat_sim.geo.Ray(obj_trans, obj_to_ee_vec)
        )
        ee_ray = ray_result.hits[0].point
        return (
            np.linalg.norm(ee_pos - ee_ray) < self._can_pick_for_ray_threshold
        )

    def force_apply_skill(self, observations, skill_name, env, obj_trans):
        # TODO: there is a bit of repeated code here. Would be better to pack the full fetch state into a high level policy
        # that can be called on different observations
        if skill_name == "nav_to_obj_waypoint":
            filtered_skill_name = "nav_to_obj"
        elif skill_name == "nav_to_robot_waypoint":
            filtered_skill_name = "nav_to_robot"
        else:
            filtered_skill_name = skill_name

        skill_walk = self._agent.actor_critic._skills[  # type: ignore
            self._agent.actor_critic._name_to_idx[filtered_skill_name]  # type: ignore
        ]

        policy_input = self._policy_info["ll_policy"]
        policy_input["observations"] = self._batch_and_apply_transforms(
            [observations]
        )

        # TODO: the observation transformation always picks up
        # the first object, we overwrite the observation
        # here, which is a bit hacky
        if skill_name == "pick":
            global_T = env._sim.get_agent_data(
                self._agent_idx
            ).articulated_agent.ee_transform()
            T_inv = global_T.inverted()
            obj_start_pos = np.array(T_inv.transform_point(obj_trans))[
                None, ...
            ]
            policy_input["observations"]["obj_start_sensor"] = obj_start_pos
        elif skill_name == "place":
            global_T = env._sim.get_agent_data(
                self._agent_idx
            ).articulated_agent.ee_transform()
            T_inv = global_T.inverted()
            obj_goal_pos = np.array(T_inv.transform_point(obj_trans))[
                None, ...
            ]
            if self._use_pick_skill_as_place_skill:
                # Reporpose pick skill for place
                policy_input["observations"]["obj_start_sensor"] = obj_goal_pos
            else:
                policy_input["observations"]["obj_goal_sensor"] = obj_goal_pos

        # Only take the goal object
        rho, phi = self.get_cartesian_obj_coords(obj_trans)
        pos_sensor = np.array([rho, -phi])[None, ...]
        policy_input["observations"]["obj_start_gps_compass"] = pos_sensor
        with torch.no_grad():
            action_data = skill_walk.act(**policy_input)
        policy_input["rnn_hidden_states"] = action_data.rnn_hidden_states
        policy_input["prev_actions"] = action_data.actions

        # Increase the step counter
        self._skill_steps += 1

        return action_data.actions

    def _is_robot_beg_motion_done(self, env, act_space):
        """Generate robot begging motion"""

        steps_body_motion = 10
        beg_lin_vel = 0.0
        beg_pitch_vel = 10.0
        animate_front_leg = 0.0
        action_array = np.zeros(get_num_actions(act_space))
        action_ind_motion = find_action_range(
            act_space, "agent_0_motion_control"
        )
        if self._skill_steps < steps_body_motion:
            # Start to beg
            action_array[
                action_ind_motion[0] : action_ind_motion[1]
            ] = np.array([beg_lin_vel, beg_pitch_vel, animate_front_leg])

        elif self._skill_steps >= TOTAL_BEG_MOTION - steps_body_motion:
            # End of begging
            action_array[
                action_ind_motion[0] : action_ind_motion[1]
            ] = np.array([beg_lin_vel, -beg_pitch_vel, animate_front_leg])
        else:
            # Animate the leg
            animate_front_leg = 1.0
            action_array[
                action_ind_motion[0] : action_ind_motion[1]
            ] = np.array([beg_lin_vel, 0.0, animate_front_leg])

        if self._skill_steps < TOTAL_BEG_MOTION:
            return False, action_array
        else:
            # Since action_array is consumed after this act function,
            # we need to return zero action array to avoid
            # height issue
            return True, np.zeros(get_num_actions(act_space))

    def _set_height(self):
        """Set the height of the robot"""
        # Get the current transformation
        trans = self.get_articulated_agent().sim_obj.transformation
        # Get the current rigid state
        rigid_state = habitat_sim.RigidState(
            mn.Quaternion.from_matrix(trans.rotation()), trans.translation
        )
        end_pos = rigid_state.translation
        end_pos[1] = ROBOT_BASE_HEIGHT
        # Get the traget transformation based on the target rigid state
        target_trans = mn.Matrix4.from_(
            rigid_state.rotation.to_matrix(),
            end_pos,
        )
        # Update the base
        self.get_articulated_agent().sim_obj.transformation = target_trans

    def _is_human_giving_a_way_to_robot(self, human_pos, robot_pos, obj_pos):
        """Check if the human blocks the robot when the robot is near the target"""
        return (
            float(np.linalg.norm(np.array((human_pos - obj_pos))[[0, 2]]))
            > START_FETCH_OBJ_DIS_THRESHOLD
            or float(np.linalg.norm(np.array((human_pos - robot_pos))[[0, 2]]))
            > START_FETCH_ROBOT_DIS_THRESHOLD
        )

    def _is_ok_to_start_fetch(self, human_pos, robot_pos, obj_pos):
        """Start to fetch the object condition"""
        return np.linalg.norm(
            self.rigid_obj_interest.linear_velocity
        ) < START_FETCH_OBJ_VEL_THRESHOLD and self._is_human_giving_a_way_to_robot(
            human_pos, robot_pos, obj_pos
        )

    def act(self, obs, env):
        # hack: assume we want to navigate to agent (1 - self._agent_idx)
        human_pos = env._sim.agents_mgr[
            1 - self._agent_idx
        ].articulated_agent.base_transformation.translation
        robot_pos = env._sim.agents_mgr[
            self._agent_idx
        ].articulated_agent.base_transformation.translation

        act_space = ActionSpace(
            {
                action_name: space
                for action_name, space in env.action_space.items()
                if "agent_0" in action_name
            }
        )
        action_array = np.zeros(get_num_actions(act_space))

        # If the robot is in the begging state, we lock it
        if self._beg_state_lock:
            self.current_state = FetchState.BEG_RESET

        if self.current_state == FetchState.WAIT:
            self.current_state = FetchState.FOLLOW
            self._init_policy_input()

        if self.current_state == FetchState.FOLLOW:
            # This is the following state, in which the robot tries to follow the human
            # There are two cases here. When the human is far away from the robot, we use
            # normal point nav, and when the human is close enough, we switch to the
            # social nav. This allows us to effectively find and follow the human
            if self.should_start_skill:
                human_robot_geo_dis = self._habitat_env._sim.geodesic_distance(
                    robot_pos, human_pos
                )

                self.start_skill(obs, "nav_to_robot")

            # First get the geo distance between the robot and the human and check if the robot can see human
            human_robot_geo_dis = self._habitat_env._sim.geodesic_distance(
                robot_pos, human_pos
            )
            human_in_frame = self._batch_and_apply_transforms([obs])[
                "humanoid_detector_sensor"
            ]
            if (
                human_robot_geo_dis
                > FOLLOW_SWITCH_GEO_DIS_FOR_POINT_SOCIAL_NAV
                and not bool(human_in_frame)
            ):
                target_pddl_skill = "nav_to_obj"
                # Robot is too far away from the human and robot cannot see the human, we should use point nav.
                # Here we init the point nav policy
                if (
                    self._follow_state_policy_chosen == FollowStatePolicy.IDLE
                    or self._follow_state_policy_chosen
                    == FollowStatePolicy.SOCIAL_NAV
                ):
                    self.start_skill(obs, target_pddl_skill)
                    self._follow_state_policy_chosen = (
                        FollowStatePolicy.POINT_NAV
                    )
                type_of_skill = self.defined_skills.nav_to_obj.skill_name
            else:
                target_pddl_skill = "nav_to_robot"
                # Robot is close enough or robot can see human, we should use social nav.
                # Here we init the social nav policy
                if (
                    self._follow_state_policy_chosen == FollowStatePolicy.IDLE
                    or self._follow_state_policy_chosen
                    == FollowStatePolicy.POINT_NAV
                ):
                    self.start_skill(obs, target_pddl_skill)
                    self._follow_state_policy_chosen = (
                        FollowStatePolicy.SOCIAL_NAV
                    )
                type_of_skill = self.defined_skills.nav_to_robot.skill_name

            if type_of_skill == "OracleNavPolicy":
                finished_nav = obs["agent_0_has_finished_oracle_nav"]

            if type_of_skill == "NavSkillPolicy":
                action_array = self.force_apply_skill(
                    obs, target_pddl_skill, env, human_pos
                )[0]

            elif type_of_skill == "OracleNavPolicy":
                action_ind_nav = find_action_range(
                    act_space, "agent_0_oracle_nav_action"
                )
                action_array[
                    action_ind_nav[0] : action_ind_nav[0] + 3
                ] = human_pos
            else:
                raise ValueError(f"Skill {type_of_skill} not recognized.")

        elif self.current_state == FetchState.SEARCH:
            if self.should_start_skill:
                # TODO: obs can be batched before
                self.start_skill(obs, "nav_to_obj")
                # We init the policy here since FOLLOW state is followed by SEARCH state
                self._init_policy_input()

            obj_trans = self.rigid_obj_interest.translation

            # Get gripper height
            ee_height = (
                self.get_articulated_agent().ee_transform().translation[1]
            )

            # Get the navigable point that is in the navmesh and the location is not unoccluded
            safe_trans = navmesh_utils.unoccluded_navmesh_snap(
                pos=obj_trans,
                height=ee_height,
                pathfinder=env._sim.pathfinder,
                sim=env._sim,
                island_id=env._sim.largest_island_idx,
                target_object_id=self.object_interest_id,
            )

            # Assign safe_trans here for the visualization
            self.safe_pos = safe_trans

            if self._is_ok_to_start_fetch(human_pos, robot_pos, obj_trans):
                type_of_skill = self.defined_skills.nav_to_obj.skill_name
                max_skill_steps = (
                    self.defined_skills.nav_to_obj.max_skill_steps
                )
                finished_nav = True
                step_terminate = False
                is_accessible = False
                is_occluded = True
                if type_of_skill == "OracleNavPolicy":
                    finished_nav = obs["agent_0_has_finished_oracle_nav"]
                else:
                    step_terminate = self._skill_steps >= max_skill_steps

                    # Make sure that there is a safe snap point
                    if safe_trans is not None:
                        target_trans = safe_trans
                        is_occluded = False
                    else:
                        # At least find snap point that is near the target obj_trans
                        search_radius = IS_ACCESSIBLE_THRESHOLD
                        candidate_trans = np.array([float("nan")] * 3)
                        # The following loop should try at most this amount
                        max_tries = 25
                        num_tries = 0
                        while (
                            np.isnan(candidate_trans).any()
                            and num_tries < max_tries
                        ):
                            # The return of pathfinder is an array type
                            candidate_trans = env._sim.pathfinder.get_random_navigable_point_near(
                                circle_center=obj_trans,
                                radius=search_radius,
                                island_index=env._sim.largest_island_idx,
                            )
                            # Increase the search radius by this amount
                            search_radius += 0.5
                            num_tries += 1

                        if num_tries >= max_tries:
                            # If we do not find a valid point within this many iterations,
                            # we then just use the current human location
                            target_trans = human_pos
                        else:
                            target_trans = candidate_trans
                        is_occluded = True
                        self.safe_pos = target_trans

                    # Check if the distance to the target safe point
                    rho, _ = self.get_cartesian_obj_coords(target_trans)
                    # Check if the agent can go there to pick up the object
                    # Ideally, we do not want the distance being too far away from the target
                    is_accessible = (
                        float(
                            np.linalg.norm(
                                np.array((target_trans - obj_trans))[[0, 2]]
                            )
                        )
                        < IS_ACCESSIBLE_THRESHOLD
                    )
                    # Finalize it. And the agent is still navigate to there even if it is not accessible
                    finished_nav = (
                        rho < self._pick_dist_threshold
                    ) or step_terminate

                if not finished_nav:
                    if type_of_skill == "NavSkillPolicy":
                        action_array = self.force_apply_skill(
                            obs, "nav_to_obj", env, target_trans
                        )[0]
                    elif type_of_skill == "OracleNavPolicy":
                        action_ind_nav = find_action_range(
                            act_space, "agent_0_oracle_nav_action"
                        )
                        action_array[
                            action_ind_nav[0] : action_ind_nav[0] + 3
                        ] = obj_trans
                    else:
                        raise ValueError(
                            f"Skill {type_of_skill} not recognized."
                        )

                else:
                    if step_terminate:
                        self.current_state = FetchState.SEARCH_TIMEOUT_WAIT
                    elif not is_accessible or is_occluded:
                        self.current_state = FetchState.BEG_RESET
                    else:
                        self.current_state = FetchState.PICK
                    self._init_policy_input()

            # Check if the human blocks the robot when robot is near the target
            self.human_block_robot_when_searching = (
                not self._is_human_giving_a_way_to_robot(
                    human_pos, robot_pos, obj_trans
                )
            )

        elif self.current_state == FetchState.SEARCH_ORACLE_NAV:
            if self.should_start_skill:
                # TODO: obs can be batched before
                self.start_skill(obs, "nav_to_obj")
            obj_trans = self.rigid_obj_interest.translation

            # Get gripper height
            ee_height = (
                self.get_articulated_agent().ee_transform().translation[1]
            )

            # Get the navigable point that is in the navmesh and the location is not unoccluded
            safe_trans = navmesh_utils.unoccluded_navmesh_snap(
                pos=obj_trans,
                height=ee_height,
                pathfinder=env._sim.pathfinder,
                sim=env._sim,
                island_id=env._sim.largest_island_idx,
                target_object_id=self.object_interest_id,
            )
            self.safe_pos = safe_trans

            if self._is_ok_to_start_fetch(human_pos, robot_pos, obj_trans):
                # Check the termination conditions
                finished_nav = True
                # Make sure that there is a safe snap point
                if safe_trans is not None:
                    target_trans = safe_trans
                else:
                    # At least find snap point that is near the target obj_trans
                    search_radius = IS_ACCESSIBLE_THRESHOLD
                    candidate_trans = np.array([float("nan")] * 3)
                    # The following loop should try at most this amount
                    max_tries = 25
                    num_tries = 0
                    while (
                        np.isnan(candidate_trans).any()
                        and num_tries < max_tries
                    ):
                        # The return of pathfinder is an array type
                        candidate_trans = env._sim.pathfinder.get_random_navigable_point_near(
                            circle_center=obj_trans,
                            radius=search_radius,
                            island_index=env._sim.largest_island_idx,
                        )
                        # Increase the search radius by this amount
                        search_radius += 0.5
                        num_tries += 1

                    if num_tries >= max_tries:
                        # If we do not find a valid point within this many iterations,
                        # we then just use the current human location
                        target_trans = human_pos
                    else:
                        target_trans = candidate_trans

                    self.safe_pos = target_trans

                rho, _ = self.get_cartesian_obj_coords(target_trans)
                is_accessible = (
                    float(
                        np.linalg.norm(
                            np.array((target_trans - obj_trans))[[0, 2]]
                        )
                    )
                    < IS_ACCESSIBLE_THRESHOLD
                )
                finished_nav = rho < self._pick_dist_threshold

                if not finished_nav:
                    action_ind_nav = find_action_range(
                        act_space, "agent_0_oracle_nav_action"
                    )
                    action_array[
                        action_ind_nav[0] : action_ind_nav[0] + 3
                    ] = target_trans
                    self.gt_path = env.task.actions[
                        "agent_0_oracle_nav_action"
                    ]._path_to_point(target_trans)
                else:
                    if not is_accessible:
                        self.current_state = FetchState.BEG_RESET
                    else:
                        self.current_state = FetchState.PICK
                    self._init_policy_input()

            # Check if the human blocks the robot when robot is near the target
            self.human_block_robot_when_searching = (
                not self._is_human_giving_a_way_to_robot(
                    human_pos, robot_pos, obj_trans
                )
            )

        elif self.current_state == FetchState.SEARCH_TIMEOUT_WAIT:
            if self.should_start_skill:
                # TODO: obs can be batched before
                self.start_skill(obs, "nav_to_obj")

        elif self.current_state == FetchState.PICK:
            obj_trans = self.rigid_obj_interest.translation
            type_of_skill = self.defined_skills.pick.skill_name
            max_skill_steps = self.defined_skills.pick.max_skill_steps
            if type_of_skill == "PickSkillPolicy":
                if self.should_start_skill:
                    # TODO: obs can be batched before
                    self.start_skill(obs, "pick")
                    # TODO: Pre-set the target object id
                    env.task.actions[
                        "agent_0_arm_action"
                    ].grip_ctrlr.pick_object_pos = (
                        self.rigid_obj_interest.translation
                    )
                action_array = self.force_apply_skill(
                    obs, "pick", env, obj_trans
                )[0]
                if self.check_if_skill_done(obs, "pick"):
                    self.rigid_obj_interest.motion_type = MotionType.KINEMATIC
                    self.grasped_object = self.rigid_obj_interest
                    self.grasped_object_id = self.object_interest_id
                    self.grasped_object.override_collision_group(
                        self._thrown_object_collision_group
                    )
                    self.current_state = FetchState.BRING
                if self._skill_steps >= max_skill_steps:
                    self.cancel_fetch()
                    # self._get_grasp_mgr(env).desnap()
                    # self.grasped_object_id = None
                    # self.grasped_object = None
                    # self.current_state = FetchState.RESET_ARM_BEFORE_WAIT
            else:
                if self.counter_pick < PICK_STEPS:
                    self.counter_pick += 1
                else:
                    self.rigid_obj_interest.motion_type = MotionType.KINEMATIC

                    assert self.object_interest_id is not None
                    self._get_grasp_mgr(env).snap_to_obj(
                        self.object_interest_id
                    )
                    self.grasped_object = self.rigid_obj_interest
                    self.grasped_object_id = self.object_interest_id
                    self.grasped_object.override_collision_group(
                        self._thrown_object_collision_group
                    )

                    self.counter_pick = 0
                    self.current_state = FetchState.BRING

        elif self.current_state == FetchState.BRING:
            if self.should_start_skill:
                # TODO: obs can be batched before
                self.start_skill(obs, "nav_to_obj")
            type_of_skill = self.defined_skills.nav_to_obj.skill_name
            max_skill_steps = self.defined_skills.nav_to_obj.max_skill_steps
            step_terminate = self._skill_steps >= max_skill_steps
            if type_of_skill == "OracleNavPolicy":
                finished_nav = obs["agent_0_has_finished_oracle_nav"]
            else:
                rho, _ = self.get_cartesian_obj_coords(human_pos)
                finished_nav = (
                    rho < self._drop_dist_threshold or step_terminate
                )

            if not finished_nav:
                # Keep gripper closed
                if type_of_skill == "NavSkillPolicy":
                    if self.should_start_skill:
                        # TODO: obs can be batched before
                        self.start_skill(obs, "nav_to_obj")
                    action_array = self.force_apply_skill(
                        obs, "nav_to_obj", env, human_pos
                    )[0]

                elif type_of_skill == "OracleNavPolicy":
                    action_ind_nav = find_action_range(
                        act_space, "agent_0_oracle_nav_action"
                    )
                    action_array[
                        action_ind_nav[0] : action_ind_nav[0] + 3
                    ] = obj_trans
                else:
                    raise ValueError(f"Skill {type_of_skill} not recognized.")

            else:
                if step_terminate:
                    self.current_state = FetchState.BRING_TIMEOUT_WAIT
                else:
                    self.current_state = FetchState.DROP
                self._init_policy_input()

        elif self.current_state == FetchState.BRING_ORACLE_NAV:
            if self.should_start_skill:
                # TODO: obs can be batched before
                self.start_skill(obs, "nav_to_obj")

            # Determinate the terminatin condition
            rho, _ = self.get_cartesian_obj_coords(human_pos)
            finished_nav = rho < self._drop_dist_threshold
            if not finished_nav:
                action_ind_nav = find_action_range(
                    act_space, "agent_0_oracle_nav_action"
                )
                action_array[
                    action_ind_nav[0] : action_ind_nav[0] + 3
                ] = human_pos
                self.gt_path = env.task.actions[
                    "agent_0_oracle_nav_action"
                ]._path_to_point(human_pos)
            else:
                self.current_state = FetchState.DROP
                self._init_policy_input()

        elif self.current_state == FetchState.BRING_TIMEOUT_WAIT:
            if self.should_start_skill:
                # TODO: obs can be batched before
                self.start_skill(obs, "nav_to_obj")

        elif self.current_state == FetchState.DROP:
            type_of_skill = self.defined_skills.place.skill_name
            max_skill_steps = self.defined_skills.place.max_skill_steps
            if type_of_skill == "PlaceSkillPolicy":
                if self.should_start_skill:
                    # TODO: obs can be batched before
                    self._policy_info = self._init_policy_input()
                    self.start_skill(obs, "place")
                if self._target_place_trans is None:  # type: ignore
                    self._target_place_trans = self.get_place_loc(env, self._local_place_target)  # type: ignore
                action_array = self.force_apply_skill(
                    obs, "place", env, self._target_place_trans
                )[0]

                # Change the object motion type if the robot desnaps the object
                if not self._get_grasp_mgr(env).is_grasped:
                    self.grasped_object.motion_type = MotionType.DYNAMIC

                if self.check_if_skill_done(obs, "place"):
                    # grasped_rigid_obj = self.grasped_object
                    # obj_bb = get_aabb(self.grasped_object_id, env._sim)
                    # self._last_object_drop_info = (
                    #     grasped_rigid_obj,
                    #     max(obj_bb.size_x(), obj_bb.size_y(), obj_bb.size_z()),
                    # )
                    # self.grasped_object_id = None
                    # self.grasped_object = None
                    # self._target_place_trans = None
                    # self.current_state = FetchState.WAIT
                    self.cancel_fetch(skip_reset_arm=True)
                if self._skill_steps >= max_skill_steps:
                    # self._get_grasp_mgr(env).desnap()
                    # self.grasped_object_id = None
                    # self.grasped_object = None
                    # self._target_place_trans = None
                    # self.current_state = FetchState.RESET_ARM_BEFORE_WAIT
                    self.cancel_fetch()
            else:
                if self.counter_pick < PICK_STEPS:
                    self.counter_pick += 1
                else:
                    self.cancel_fetch()

        elif self.current_state == FetchState.BEG_RESET:
            # We only restart the beg_reset state if the current beg_reset state finished
            if self.should_start_skill and not self._beg_state_lock:
                self.start_skill(obs, "nav_to_obj")
                self._beg_state_lock = True
                self._robot_obj_T = mn.Matrix4(
                    self.get_articulated_agent().sim_obj.transformation
                )

            # Since we do not apply the skill, so we need to incease the step counter
            self._skill_steps += 1
            is_done, action_array = self._is_robot_beg_motion_done(
                env, act_space
            )
            if is_done:
                self._beg_state_lock = False
                self.get_articulated_agent().sim_obj.transformation = (
                    mn.Matrix4(self._robot_obj_T)
                )
                self._robot_obj_T = None
                self.cancel_fetch()
                # self.current_state = FetchState.WAIT

        elif self.current_state == FetchState.RESET_ARM_BEFORE_WAIT:
            type_of_skill = self.defined_skills.place.skill_name

            if type_of_skill == "PlaceSkillPolicy":
                if self.should_start_skill:
                    # TODO: obs can be batched before
                    self._policy_info = self._init_policy_input()
                    self.start_skill(obs, "place")
                if self._target_place_trans is None:  # type: ignore
                    self._target_place_trans = self.get_ee_loc(env)  # type: ignore
                action_array = self.force_apply_skill(
                    obs, "place", env, self._target_place_trans
                )[0]
                if self.check_if_skill_done(obs, "place"):
                    self._target_place_trans = None
                    self.current_state = FetchState.WAIT
            else:
                if self.counter_pick < PICK_STEPS:
                    self.counter_pick += 1
                else:
                    self.cancel_fetch()

        if self._last_object_drop_info is not None:
            grasp_mgr = self._get_grasp_mgr(env)

            # when the thrown object leaves the hand, update the collisiongroups
            rigid_obj = self._last_object_drop_info[0]
            ee_pos = (
                self.get_articulated_agent()
                .ee_transform(grasp_mgr.ee_index)
                .translation
            )
            dist = np.linalg.norm(ee_pos - rigid_obj.translation)
            if dist >= self._last_object_drop_info[1]:
                # rigid_obj.override_collision_group(CollisionGroups.Default)
                self._last_object_drop_info = None

        # Make sure the height is consistents
        self._set_height()

        return action_array

    def on_environment_reset(self):
        self._step_i = 0
        self.current_state = FetchState.WAIT
        self.object_interest_id = None
        self.rigid_obj_interest = None
        self.grasped_object_id = None
        self.grasped_object = None
        self._last_object_drop_info = None
        self.counter_pick = 0
        self._policy_info = self._init_policy_input()
        self._target_place_trans = None
        self._skill_steps = 0
        self._beg_state_lock = False
        self._robot_obj_T = None
        self.safe_pos = None
        self.gt_path = None
        self.human_block_robot_when_searching = False
