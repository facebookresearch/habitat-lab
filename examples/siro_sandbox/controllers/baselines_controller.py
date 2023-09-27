#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import TYPE_CHECKING

import gym.spaces as spaces
import numpy as np
import torch

import habitat.gym.gym_wrapper as gym_wrapper
from habitat.core.spaces import ActionSpace
from habitat.tasks.rearrange.utils import get_aabb
from habitat.tasks.utils import cartesian_to_polar
from habitat_baselines.common.env_spec import EnvironmentSpec
from habitat_baselines.common.obs_transformers import get_active_obs_transforms
from habitat_baselines.rl.hrl.utils import find_action_range
from habitat_baselines.rl.multi_agent.multi_agent_access_mgr import (
    MultiAgentAccessMgr,
)
from habitat_baselines.rl.multi_agent.utils import (
    update_dict_with_agent_prefix,
)
from habitat_baselines.rl.ppo.single_agent_access_mgr import (
    SingleAgentAccessMgr,
)
from habitat_baselines.utils.common import get_num_actions
from habitat_sim.physics import CollisionGroups, MotionType

from .controller_abc import BaselinesController

if TYPE_CHECKING:
    from omegaconf import DictConfig

    import habitat
    from habitat.core.environments import GymHabitatEnv


def clean_dict(d, remove_prefix):
    ret_d = {}
    for k, v in d.spaces.items():
        if k.startswith(remove_prefix):
            new_k = k[len(remove_prefix) :]
            if isinstance(v, spaces.Dict):
                ret_d[new_k] = clean_dict(v, remove_prefix)
            else:
                ret_d[new_k] = v
        elif not k.startswith("agent"):
            ret_d[k] = v
    return spaces.Dict(ret_d)


class SingleAgentBaselinesController(BaselinesController):
    """Controller for single baseline agent."""

    def __init__(
        self,
        agent_idx: int,
        is_multi_agent: bool,
        config: "DictConfig",
        gym_habitat_env: "GymHabitatEnv",
    ):
        self._agent_idx: int = agent_idx
        self._agent_name: str = config.habitat.simulator.agents_order[
            self._agent_idx
        ]

        self._agent_k: str
        if is_multi_agent:
            self._agent_k = f"agent_{self._agent_idx}_"
        else:
            self._agent_k = ""

        super().__init__(
            is_multi_agent,
            config,
            gym_habitat_env,
        )

    def _create_env_spec(self):
        # udjust the observation and action space to be agent specific (remove other agents)
        original_action_space = clean_dict(
            self._gym_habitat_env.original_action_space, self._agent_k
        )
        observation_space = clean_dict(
            self._gym_habitat_env.observation_space, self._agent_k
        )
        action_space = gym_wrapper.create_action_space(original_action_space)

        env_spec = EnvironmentSpec(
            observation_space=observation_space,
            action_space=action_space,
            orig_action_space=original_action_space,
        )

        return env_spec

    def _get_active_obs_transforms(self):
        return get_active_obs_transforms(self._config, self._agent_name)

    def _create_agent(self):
        agent = SingleAgentAccessMgr(
            agent_name=self._agent_name,
            config=self._config,
            env_spec=self._env_spec,
            num_envs=self._num_envs,
            is_distrib=False,
            device=self.device,
            percent_done_fn=lambda: 0,
        )

        return agent

    def _load_agent_state_dict(self, checkpoint):
        self._agent.load_state_dict(checkpoint[self._agent_idx])

    def _batch_and_apply_transforms(self, obs):
        batch = super()._batch_and_apply_transforms(obs)
        batch = update_dict_with_agent_prefix(batch, self._agent_idx)

        return batch


class MultiAgentBaselinesController(BaselinesController):
    """Controller for multiple baseline agents."""

    def _create_env_spec(self):
        observation_space = self._gym_habitat_env.observation_space
        action_space = self._gym_habitat_env.action_space
        original_action_space = self._gym_habitat_env.original_action_space

        env_spec = EnvironmentSpec(
            observation_space=observation_space,
            action_space=action_space,
            orig_action_space=original_action_space,
        )

        return env_spec

    def _get_active_obs_transforms(self):
        return get_active_obs_transforms(self._config)

    def _create_agent(self):
        agent = MultiAgentAccessMgr(
            config=self._config,
            env_spec=self._env_spec,
            num_envs=self._num_envs,
            is_distrib=False,
            device=self.device,
            percent_done_fn=lambda: 0,
        )

        return agent

    def _load_agent_state_dict(self, checkpoint):
        self._agent.load_state_dict(checkpoint)


class FetchState(Enum):
    WAIT = 1
    SEARCH = 2
    PICK = 3
    BRING = 4
    DROP = 5
    RESET_ARM_BEFORE_WAIT = 6


PICK_STEPS = 40


class FetchBaselinesController(SingleAgentBaselinesController):
    def __init__(
        self, agent_idx, is_multi_agent, config, gym_env, habitat_env
    ):
        self._current_state = FetchState.WAIT
        self.should_start_skill = False
        self.object_interest_id = None
        self.rigid_obj_interest = None
        self.grasped_object_id = None
        self.grasped_object = None
        self._last_object_drop_info = None
        self._env = gym_env
        self._thrown_object_collision_group = CollisionGroups.UserGroup7
        self.counter_pick = 0
        self._habitat_env: habitat.Env = habitat_env  # type: ignore
        # also consider self._config.habitat.task["robot_at_thresh"]
        self._pick_dist_threshold = 1.2
        self._drop_dist_threshold = 1.8
        # arm local ee location format: [up,right,front]
        self._local_place_target = [-0.1, 0.0, 0.5]
        super().__init__(agent_idx, is_multi_agent, config, gym_env)
        self._policy_info = self._init_policy_input()
        self.defined_skills = self._config.habitat_baselines.rl.policy[
            self._agent_name
        ].hierarchical_policy.defined_skills
        self._use_pick_skill_as_place_skill = (
            self.defined_skills.place.use_pick_skill_as_place_skill
        )

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

    def cancel_fetch(self):
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
            or self.current_state == FetchState.RESET_ARM_BEFORE_WAIT
        ):
            self.current_state = FetchState.RESET_ARM_BEFORE_WAIT
        else:
            self.current_state = FetchState.WAIT
        self.counter_pick = 0
        self.object_interest_id = None
        self.rigid_obj_interest = None
        self._target_place_trans = None

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

    def force_apply_skill(self, observations, skill_name, env, obj_trans):
        # TODO: there is a bit of repeated code here. Would be better to pack the full fetch state into a high level policy
        # that can be called on different observations
        skill_walk = self._agent.actor_critic._skills[  # type: ignore
            self._agent.actor_critic._name_to_idx[skill_name]  # type: ignore
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
        rho, phi = self.get_geodesic_distance_obj_coords(obj_trans)
        pos_sensor = np.array([rho, -phi])[None, ...]
        policy_input["observations"]["obj_start_gps_compass"] = pos_sensor
        with torch.no_grad():
            action_data = skill_walk.act(**policy_input)
        policy_input["rnn_hidden_states"] = action_data.rnn_hidden_states
        policy_input["prev_actions"] = action_data.actions
        return action_data.actions

    def act(self, obs, env):
        # hack: assume we want to navigate to agent (1 - self._agent_idx)
        human_trans = env._sim.agents_mgr[
            1 - self._agent_idx
        ].articulated_agent.base_transformation.translation
        act_space = ActionSpace(
            {
                action_name: space
                for action_name, space in env.action_space.items()
                if "agent_0" in action_name
            }
        )
        action_array = np.zeros(get_num_actions(act_space))

        if self.current_state == FetchState.SEARCH:
            if self.should_start_skill:
                # TODO: obs can be batched before
                self.start_skill(obs, "nav_to_obj")
            obj_trans = self.rigid_obj_interest.translation

            if np.linalg.norm(self.rigid_obj_interest.linear_velocity) < 1.5:
                type_of_skill = self.defined_skills.nav_to_obj.skill_name

                if type_of_skill == "OracleNavPolicy":
                    finished_nav = obs["agent_0_has_finished_oracle_nav"]
                else:
                    # agent_trans = human_trans
                    rho, _ = self.get_cartesian_obj_coords(obj_trans)
                    finished_nav = rho < self._pick_dist_threshold

                if not finished_nav:
                    if type_of_skill == "NavSkillPolicy":
                        action_array = self.force_apply_skill(
                            obs, "nav_to_obj", env, obj_trans
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
                    # Grab object
                    self._init_policy_input()

                    self.current_state = FetchState.PICK

        elif self.current_state == FetchState.PICK:
            obj_trans = self.rigid_obj_interest.translation
            type_of_skill = self.defined_skills.pick.skill_name
            if type_of_skill == "PickSkillPolicy":
                if self.should_start_skill:
                    # TODO: obs can be batched before
                    self.start_skill(obs, "pick")
                action_array = self.force_apply_skill(
                    obs, "pick", env, obj_trans
                )[0]
                if self.check_if_skill_done(obs, "pick"):
                    self.grasped_object = self.rigid_obj_interest
                    self.grasped_object_id = self.object_interest_id
                    self.grasped_object.override_collision_group(
                        self._thrown_object_collision_group
                    )
                    self.current_state = FetchState.BRING
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
            type_of_skill = self.defined_skills.nav_to_robot.skill_name
            if type_of_skill == "OracleNavPolicy":
                finished_nav = obs["agent_0_has_finished_oracle_nav"]
            else:
                # agent_trans = human_trans
                rho, _ = self.get_cartesian_obj_coords(human_trans)
                finished_nav = rho < self._pick_dist_threshold

            if not finished_nav:
                # Keep gripper closed
                if type_of_skill == "NavSkillPolicy":
                    if self.should_start_skill:
                        # TODO: obs can be batched before
                        self.start_skill(obs, "nav_to_robot")
                    action_array = self.force_apply_skill(
                        obs, "nav_to_robot", env, human_trans
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
                self.current_state = FetchState.DROP
                self._init_policy_input()

        elif self.current_state == FetchState.DROP:
            type_of_skill = self.defined_skills.place.skill_name

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
                if self.check_if_skill_done(obs, "place"):
                    self.grasped_object.motion_type = MotionType.DYNAMIC
                    grasped_rigid_obj = self.grasped_object
                    obj_bb = get_aabb(self.grasped_object_id, env._sim)
                    self._last_object_drop_info = (
                        grasped_rigid_obj,
                        max(obj_bb.size_x(), obj_bb.size_y(), obj_bb.size_z()),
                    )
                    self.grasped_object_id = None
                    self.grasped_object = None
                    self._target_place_trans = None
                    self.current_state = FetchState.WAIT
            else:
                if self.counter_pick < PICK_STEPS:
                    self.counter_pick += 1
                else:
                    self.cancel_fetch()

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
