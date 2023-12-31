#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
import torch

import habitat
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
)
from habitat_baselines.utils.common import (
    batch_obs,
    get_action_space_info,
    is_continuous_action_space,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from habitat.core.environments import GymHabitatEnv
    from habitat_baselines.common.env_spec import EnvironmentSpec
    from habitat_baselines.common.obs_transformers import (
        ObservationTransformer,
    )
    from habitat_baselines.rl.ppo.agent_access_mgr import AgentAccessMgr


class Controller(ABC):
    """Abstract controller."""

    def __init__(self, is_multi_agent):
        self._is_multi_agent = is_multi_agent

    @abstractmethod
    def act(self, obs, env):
        pass

    def on_environment_reset(self):
        pass


class GuiController(Controller):
    """Abstract controller for gui agents."""

    def __init__(self, agent_idx, is_multi_agent, gui_input):
        super().__init__(is_multi_agent)
        self._agent_idx = agent_idx
        self._gui_input = gui_input


class BaselinesController(Controller):
    """Abstract controller for baselines agents."""

    def __init__(
        self,
        is_multi_agent: bool,
        config: "DictConfig",
        gym_habitat_env: "GymHabitatEnv",
    ):
        super().__init__(is_multi_agent)
        self._config: DictConfig = config

        self._gym_habitat_env: GymHabitatEnv = gym_habitat_env
        self._habitat_env: habitat.Env = gym_habitat_env.unwrapped.habitat_env
        self._num_envs: int = 1

        self.device: torch.device = (
            torch.device("cuda", config.habitat_baselines.torch_gpu_id)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # create env spec
        self._env_spec: EnvironmentSpec = self._create_env_spec()

        # create observations transforms
        self._obs_transforms: List[
            "ObservationTransformer"
        ] = self._get_active_obs_transforms()

        # apply observations transforms
        self._env_spec.observation_space = apply_obs_transforms_obs_space(
            self._env_spec.observation_space, self._obs_transforms
        )

        # create agent
        self._agent: "AgentAccessMgr" = self._create_agent()
        if (
            self._agent.actor_critic.should_load_agent_state
            and self._config.habitat_baselines.eval.should_load_ckpt
        ):
            self._load_agent_checkpoint()

        self._agent.eval()

        self._action_shape: Tuple[int]
        self._discrete_actions: bool
        self._action_shape, self._discrete_actions = get_action_space_info(
            self._agent.actor_critic.policy_action_space
        )

        hidden_state_lens = self._agent.actor_critic.hidden_state_shape_lens
        action_space_lens = (
            self._agent.actor_critic.policy_action_space_shape_lens
        )

        self._space_lengths: Dict = {}
        n_agents = len(self._config.habitat.simulator.agents)
        if n_agents > 1:
            self._space_lengths = {
                "index_len_recurrent_hidden_states": hidden_state_lens,
                "index_len_prev_actions": action_space_lens,
            }

        # these attributes are used for inference
        # and will be set in on_environment_reset
        self._test_recurrent_hidden_states = None
        self._prev_actions = None
        self._not_done_masks = None

    @abstractmethod
    def _create_env_spec(self):
        pass

    @abstractmethod
    def _get_active_obs_transforms(self):
        pass

    @abstractmethod
    def _create_agent(self):
        pass

    @abstractmethod
    def _load_agent_state_dict(self, checkpoint):
        pass

    def _load_agent_checkpoint(self):
        checkpoint = torch.load(
            self._config.habitat_baselines.eval_ckpt_path_dir,
            map_location="cpu",
        )
        self._load_agent_state_dict(checkpoint)

    def _batch_and_apply_transforms(self, obs):
        batch = batch_obs(obs, device=self.device)
        batch = apply_obs_transforms_batch(batch, self._obs_transforms)

        return batch

    def on_environment_reset(self):
        self._test_recurrent_hidden_states = torch.zeros(
            (
                self._num_envs,
                *self._agent.actor_critic.hidden_state_shape,
            ),
            device=self.device,
        )

        self._prev_actions = torch.zeros(
            self._num_envs,
            *self._action_shape,
            device=self.device,
            dtype=torch.long if self._discrete_actions else torch.float,
        )

        self._not_done_masks = torch.zeros(
            (
                self._num_envs,
                *self._agent.masks_shape,
            ),
            device=self.device,
            dtype=torch.bool,
        )

    def act(self, obs, env):
        batch = self._batch_and_apply_transforms([obs])

        with torch.no_grad():
            action_data = self._agent.actor_critic.act(
                batch,
                self._test_recurrent_hidden_states,
                self._prev_actions,
                self._not_done_masks,
                deterministic=False,
                **self._space_lengths,
            )
            if action_data.should_inserts is None:
                self._test_recurrent_hidden_states = (
                    action_data.rnn_hidden_states
                )
                self._prev_actions.copy_(action_data.actions)  # type: ignore
            else:
                self._agent.actor_critic.update_hidden_state(
                    self._test_recurrent_hidden_states,
                    self._prev_actions,
                    action_data,
                )

        assert len(action_data.env_actions) == 1
        if is_continuous_action_space(self._env_spec.action_space):
            # Clipping actions to the specified limits
            action = np.clip(
                action_data.env_actions[0].cpu().numpy(),
                self._env_spec.action_space.low,
                self._env_spec.action_space.high,
            )
        else:
            action = action_data.env_actions[0].cpu().item()

        # _not_done_masks serves as en indicator of whether the episode is done
        # it is reset to False in on_environment_reset
        self._not_done_masks.fill_(True)  # type: ignore [attr-defined]

        return action
