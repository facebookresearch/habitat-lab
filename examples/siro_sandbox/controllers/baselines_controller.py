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
from habitat_sim.physics import CollisionGroups

from .controller_abc import BaselinesController

if TYPE_CHECKING:
    from omegaconf import DictConfig

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


PICK_STEPS = 20


class FetchBaselinesController(SingleAgentBaselinesController):
    def __init__(
        self,
        agent_idx,
        is_multi_agent,
        config,
        env,
    ):
        self._current_state = FetchState.WAIT
        self.should_start_skill = False
        self.object_interest_id = None
        self.rigid_obj_interest = None
        self.grasped_object_id = None
        self.grasped_object = None
        self._last_object_drop_info = None
        self._env = env
        self._thrown_object_collision_group = CollisionGroups.UserGroup7
        self.counter_pick = 0

        super().__init__(agent_idx, is_multi_agent, config, env)
        self._policy_info = self._init_policy_input()
        self.defined_skills = self._config.habitat_baselines.rl.policy[
            self._agent_name
        ].hierarchical_policy.defined_skills

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

    def get_articulated_agent(self, env):
        return env._sim.agents_mgr[self._agent_idx].articulated_agent

    def start_skill(self, observations, skill_name):
        skill_walk = self._agent.actor_critic._skills[
            self._agent.actor_critic._name_to_idx[skill_name]
        ]
        policy_input = self._policy_info["ll_policy"]
        obs = self._batch_and_apply_transforms([observations])
        prev_actions = policy_input["prev_actions"]
        rnn_hidden_states = policy_input["rnn_hidden_states"]
        batch_id = 0
        if skill_name == "pick":
            skill_args = ["0|0"]
        else:
            skill_args = ["", "0|0", ""]
        (
            rnn_hidden_states[batch_id],
            prev_actions[batch_id],
        ) = skill_walk.on_enter(
            [skill_args], [batch_id], obs, rnn_hidden_states, prev_actions
        )
        self.should_start_skill = False

    def get_cartesian_obj_coords(self, obj_trans, env):
        articulated_agent_T = self.get_articulated_agent(
            env
        ).base_transformation
        rel_pos = articulated_agent_T.inverted().transform_point(obj_trans)
        rho, phi = cartesian_to_polar(rel_pos[0], rel_pos[1])

        return rho, phi

    def force_apply_skill(self, observations, skill_name, env, obj_trans):
        # TODO: there is a bit of repeated code here. Would be better to pack the full fetch state into a high level policy
        # that can be called on different observations
        skill_walk = self._agent.actor_critic._skills[
            self._agent.actor_critic._name_to_idx[skill_name]
        ]
        policy_input = self._policy_info["ll_policy"]
        policy_input["observations"] = self._batch_and_apply_transforms(
            [observations]
        )
        # Only take the goal object

        rho, phi = self.get_cartesian_obj_coords(obj_trans, env)
        pos_sensor = np.array([rho, -phi])[None, ...]
        policy_input["observations"]["obj_start_gps_compass"] = pos_sensor
        with torch.no_grad():
            action_data = skill_walk.act(**policy_input)
        policy_input["rnn_hidden_states"] = action_data.rnn_hidden_states
        policy_input["prev_actions"] = action_data.actions
        return action_data.actions

    def act(self, obs, env):
        human_trans = env._sim.agents_mgr[
            1 - self._agent_idx
        ].articulated_agent.base_transformation.translation
        finish_oracle_nav = obs["agent_0_has_finished_oracle_nav"]
        act_space = ActionSpace(
            {
                action_name: space
                for action_name, space in env.action_space.items()
                if "agent_0" in action_name
            }
        )
        action_array = np.zeros(get_num_actions(act_space))
        action_ind_nav = find_action_range(
            act_space, "agent_0_oracle_nav_action"
        )

        if self.current_state == FetchState.SEARCH:
            if self.should_start_skill:
                # TODO: obs can be batched before
                self.start_skill(obs, "nav_to_obj")
            obj_trans = self.rigid_obj_interest.translation

            if np.linalg.norm(self.rigid_obj_interest.linear_velocity) < 1.5:
                rho, _ = self.get_cartesian_obj_coords(obj_trans, env)
                type_of_skill = self.defined_skills.nav_to_obj.skill_name

                if (
                    type_of_skill != "OracleNavPolicy"
                    and rho < self._config.habitat.task["robot_at_thresh"]
                ):
                    finish_oracle_nav = True
                if not finish_oracle_nav:
                    if type_of_skill == "NavSkillPolicy":
                        action_array = self.force_apply_skill(
                            obs, "nav_to_obj", env, obj_trans
                        )[0]
                    elif type_of_skill == "OracleNavPolicy":
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
            else:
                if self.counter_pick < PICK_STEPS:
                    self.counter_pick += 1
                else:
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
            agent_trans = human_trans
            rho, _ = self.get_cartesian_obj_coords(agent_trans, env)
            if (
                type_of_skill != "OracleNavPolicy"
                and rho < self._config.habitat.task["robot_at_thresh"]
            ):
                finish_oracle_nav = True

            if not finish_oracle_nav:
                # Keep gripper closed
                if type_of_skill == "NavSkillPolicy":
                    action_array = self.force_apply_skill(
                        obs, "nav_to_robot", env, human_trans
                    )[0]
                elif type_of_skill == "OracleNavPolicy":
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
                    self.start_skill(obs, "place")
                action_array = self.force_apply_skill(
                    obs, "place", env, obj_trans
                )[0]
            else:
                if self.counter_pick < PICK_STEPS:
                    self.counter_pick += 1
                else:
                    # Open gripper
                    self._get_grasp_mgr(env).desnap()

                    grasped_rigid_obj = self.grasped_object

                    obj_bb = get_aabb(self.grasped_object_id, env._sim)
                    self._last_object_drop_info = (
                        grasped_rigid_obj,
                        max(obj_bb.size_x(), obj_bb.size_y(), obj_bb.size_z()),
                    )
                    self.grasped_object_id = None
                    self.grasped_object = None
                    self.current_state = FetchState.WAIT

                    self.counter_pick = 0

        if self._last_object_drop_info is not None:
            grasp_mgr = self._get_grasp_mgr(env)

            # when the thrown object leaves the hand, update the collisiongroups
            rigid_obj = self._last_object_drop_info[0]
            ee_pos = (
                self.get_articulated_agent(env)
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
