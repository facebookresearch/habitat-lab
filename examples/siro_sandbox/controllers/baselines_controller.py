#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import TYPE_CHECKING

import gym.spaces as spaces
import numpy as np

import habitat.gym.gym_wrapper as gym_wrapper
from habitat.core.spaces import ActionSpace
from habitat.tasks.rearrange.utils import get_aabb
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
    PICK = 2
    BRING = 3


class FetchBaselinesController(SingleAgentBaselinesController):
    def __init__(
        self,
        agent_idx,
        is_multi_agent,
        config,
        env,
    ):
        self.current_state = FetchState.WAIT
        self.object_interest_id = None
        self.rigid_obj_interest = None
        self.grasped_object_id = None
        self.grasped_object = None
        self._last_object_drop_info = None
        self._env = env
        self._thrown_object_collision_group = CollisionGroups.UserGroup7

        super().__init__(agent_idx, is_multi_agent, config, env)

    def _get_grasp_mgr(self, env):
        agents_mgr = env._sim.agents_mgr
        grasp_mgr = agents_mgr._all_agent_data[self._agent_idx].grasp_mgr
        return grasp_mgr

    def get_articulated_agent(self, env):
        return env._sim.agents_mgr[self._agent_idx].articulated_agent

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

        if self.current_state == FetchState.PICK:
            obj_trans = self.rigid_obj_interest.translation
            if not finish_oracle_nav:
                action_array[
                    action_ind_nav[0] : action_ind_nav[0] + 3
                ] = obj_trans

            else:
                self._get_grasp_mgr(env).snap_to_obj(self.object_interest_id)
                self.grasped_object = self.rigid_obj_interest
                self.grasped_object_id = self.object_interest_id
                self.grasped_object.override_collision_group(
                    self._thrown_object_collision_group
                )
                self.current_state = FetchState.BRING

        elif self.current_state == FetchState.BRING:
            if not finish_oracle_nav:
                # Keep gripper closed
                action_array[
                    action_ind_nav[0] : action_ind_nav[0] + 3
                ] = human_trans
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
                rigid_obj.override_collision_group(CollisionGroups.Default)
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
