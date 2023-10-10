# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

import gym.spaces as spaces
import numpy as np
import torch

from habitat.core.spaces import ActionSpace
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.logging import baselines_logger
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.config.default import get_config
from habitat_baselines.rl.hrl.skills.skill import SkillPolicy
from habitat_baselines.rl.ppo.policy import PolicyActionData
from habitat_baselines.utils.common import get_num_actions


def truncate_obs_space(space: spaces.Box, truncate_len: int) -> spaces.Box:
    """
    Returns an observation space with taking on the first `truncate_len` elements of the space.
    """
    return spaces.Box(
        low=space.low[..., :truncate_len],
        high=space.high[..., :truncate_len],
        dtype=np.float32,
    )


class NnSkillPolicy(SkillPolicy):
    """
    Defines a skill to be used in the TP+SRL baseline.
    """

    def __init__(
        self,
        wrap_policy,
        config,
        action_space: spaces.Space,
        filtered_obs_space: spaces.Space,
        filtered_action_space: spaces.Space,
        batch_size,
        should_keep_hold_state: bool = False,
    ):
        """
        :param action_space: The overall action space of the entire task, not task specific.
        """
        super().__init__(
            config, action_space, batch_size, should_keep_hold_state
        )
        self._wrap_policy = wrap_policy
        self._filtered_obs_space = filtered_obs_space
        self._filtered_action_space = filtered_action_space
        self._ac_start = 0
        self._ac_len = get_num_actions(filtered_action_space)
        self._did_want_done = torch.zeros(self._batch_size)

        for k, space in action_space.items():
            if k not in filtered_action_space.spaces.keys():
                self._ac_start += get_num_actions(space)
            else:
                break

        self._internal_log(
            f"Skill {self._config.skill_name}: action offset {self._ac_start}, action length {self._ac_len}"
        )

    @property
    def required_obs_keys(self):
        return super().required_obs_keys + list(
            self._filtered_obs_space.spaces.keys()
        )

    def parameters(self):
        if self._wrap_policy is not None:
            return self._wrap_policy.parameters()
        else:
            return []

    @property
    def num_recurrent_layers(self):
        if self._wrap_policy is not None:
            return self._wrap_policy.net.num_recurrent_layers
        else:
            return 0

    def to(self, device):
        super().to(device)
        self._did_want_done = self._did_want_done.to(device)
        if self._wrap_policy is not None:
            self._wrap_policy.to(device)

    def on_enter(
        self,
        skill_arg,
        batch_idxs,
        observations,
        rnn_hidden_states,
        prev_actions,
        skill_name,
    ):
        ret = super().on_enter(
            skill_arg,
            batch_idxs,
            observations,
            rnn_hidden_states,
            prev_actions,
            skill_name,
        )
        self._did_want_done *= 0.0
        return ret

    def _get_filtered_obs(self, observations, cur_batch_idx) -> TensorDict:
        return TensorDict(
            {
                k: observations[k]
                for k in self._filtered_obs_space.spaces.keys()
            }
        )

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ) -> PolicyActionData:
        filtered_obs = self._get_filtered_obs(observations, cur_batch_idx)

        filtered_prev_actions = prev_actions[
            :, self._ac_start : self._ac_start + self._ac_len
        ]
        filtered_obs = self._select_obs(filtered_obs, cur_batch_idx)

        action_data = self._wrap_policy.act(
            filtered_obs,
            rnn_hidden_states,
            filtered_prev_actions,
            masks,
            deterministic,
        )
        full_action = torch.zeros(
            (masks.shape[0], self._full_ac_size), device=masks.device
        )
        full_action[
            :, self._ac_start : self._ac_start + self._ac_len
        ] = action_data.actions
        action_data.actions = full_action

        self._did_want_done[cur_batch_idx] = full_action[
            :, self._stop_action_idx
        ]
        return action_data

    @classmethod
    def from_config(
        cls, config, observation_space, action_space, batch_size, full_config
    ):
        # Load the wrap policy from file
        if len(config.load_ckpt_file) == 0:
            ckpt_dict = {}
            policy_cfg = get_config(
                "rearrange/rl_skill.yaml",
                [config.force_config_file],
            )
        else:
            try:
                ckpt_dict = torch.load(
                    config.load_ckpt_file, map_location="cpu"
                )
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Could not load neural network weights for skill from ckpt {config.load_ckpt_file}"
                ) from e

            policy_cfg = ckpt_dict["config"]

        policy = baseline_registry.get_policy(config.name)

        expected_obs_keys = policy_cfg.habitat.gym.obs_keys
        filtered_obs_space = spaces.Dict(
            {k: observation_space.spaces[k] for k in expected_obs_keys}
        )

        for k in config.obs_skill_inputs:
            if k not in filtered_obs_space.spaces:
                raise ValueError(f"Could not find {k} for skill")
            space = filtered_obs_space.spaces[k]
            # There is always a 3D position
            filtered_obs_space.spaces[k] = truncate_obs_space(space, 3)
        baselines_logger.debug(
            f"Skill {config.skill_name}: Loaded observation space {filtered_obs_space}",
        )

        filtered_action_space = ActionSpace(
            OrderedDict(
                (k, action_space[k])
                for k in policy_cfg.habitat.task.actions.keys()
            )
        )

        if "arm_action" in filtered_action_space.spaces and (
            policy_cfg.habitat.task.actions.arm_action.grip_controller is None
        ):
            filtered_action_space["arm_action"] = spaces.Dict(
                {
                    k: v
                    for k, v in filtered_action_space["arm_action"].items()
                    if k != "grip_action"
                }
            )

        baselines_logger.debug(
            f"Loaded action space {filtered_action_space} for skill {config.skill_name}",
        )

        actor_critic = policy.from_config(
            policy_cfg, filtered_obs_space, filtered_action_space
        )
        if len(ckpt_dict) > 0:
            try:
                actor_critic.load_state_dict(ckpt_dict["state_dict"])

            except Exception as e:
                raise ValueError(
                    f"Could not load checkpoint for skill {config.skill_name} from {config.load_ckpt_file}"
                ) from e

        return cls(
            actor_critic,
            config,
            action_space,
            filtered_obs_space,
            filtered_action_space,
            batch_size,
        )
