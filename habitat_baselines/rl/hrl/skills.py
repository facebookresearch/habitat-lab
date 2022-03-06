import os.path as osp

import gym.spaces as spaces
import torch
import torch.nn as nn

from habitat.core.spaces import ActionSpace
from habitat.tasks.rearrange.sub_tasks.nav_to_obj_sensors import (
    DistToNavGoalSensor,
)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.hrl.high_level_policy import (
    GtHighLevelPolicy,
    HighLevelPolicy,
)
from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.utils.common import get_num_actions


class NnSkillPolicy(Policy):
    def __init__(
        self,
        wrap_policy,
        config,
        action_space,
        filtered_obs_space,
        filtered_action_space,
        batch_size,
    ):
        self._wrap_policy = wrap_policy
        self._action_space = action_space
        self._config = config
        self._batch_size = batch_size
        self._filtered_obs_space = filtered_obs_space
        self._filtered_action_space = filtered_action_space
        self._ac_start = 0
        self._ac_len = get_num_actions(filtered_action_space)

        self._cur_skill_args = torch.zeros(self._batch_size, dtype=torch.int32)

        for k in action_space:
            if k not in filtered_action_space.keys():
                self._ac_start += get_num_actions(action_space[k])
            else:
                break

    def _select_obs(self, obs, cur_batch_idx):
        entity_positions = obs[self._config.OBS_SKILL_INPUT].view(1, -1, 3)
        obs[self._config.OBS_SKILL_INPUT] = entity_positions[
            :, self._cur_skill_args[cur_batch_idx].item()
        ]
        return obs

    def should_terminate(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> torch.Tensor:
        return torch.zeros(observations.shape[0]).to(masks.device)

    def on_enter(self, skill_arg, new_skill_batch_idx):
        self._cur_skill_args[new_skill_batch_idx] = skill_arg

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
        if self._wrap_policy is not None:
            self._wrap_policy.to(device)

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        filtered_obs = TensorDict(
            {
                k: v
                for k, v in observations.items()
                if k in self._filtered_obs_space.keys()
            }
        )
        filtered_prev_actions = prev_actions[
            :, self._ac_start : self._ac_start + self._ac_len
        ]
        filtered_obs = self._select_obs(filtered_obs, cur_batch_idx)

        _, action, _, _ = self._wrap_policy.act(
            filtered_obs,
            rnn_hidden_states,
            filtered_prev_actions,
            masks,
            deterministic,
        )
        full_action = torch.zeros(prev_actions.shape)
        full_action[:, self._ac_start : self._ac_start + self._ac_len] = action
        return full_action

    @classmethod
    def from_config(cls, config, observation_space, action_space, batch_size):
        # Load the wrap policy from file
        if len(config.LOAD_CKPT_FILE) == 0:
            raise ValueError("Need to specify load location")

        ckpt_dict = torch.load(config.LOAD_CKPT_FILE, map_location="cpu")
        policy = baseline_registry.get_policy(config.name)
        policy_cfg = ckpt_dict["config"]

        filtered_obs_space = spaces.Dict(
            {
                k: v
                for k, v in observation_space.spaces.items()
                if (k in policy_cfg.RL.POLICY.include_visual_keys)
                or (k in policy_cfg.RL.GYM_OBS_KEYS)
            }
        )
        filtered_action_space = ActionSpace(
            {
                k: v
                for k, v in action_space.spaces.items()
                if k in policy_cfg.TASK_CONFIG.TASK.ACTIONS
            }
        )

        ###############################################
        # TEMPORARY CODE TO ADD MISSING CONFIG KEYS
        policy_cfg.defrost()
        policy_cfg.RL.POLICY.ACTION_DIST.use_std_param = False
        policy_cfg.RL.POLICY.ACTION_DIST.clamp_std = False
        policy_cfg.freeze()
        ###############################################

        actor_critic = policy.from_config(
            policy_cfg, filtered_obs_space, filtered_action_space
        )
        actor_critic.load_state_dict(
            {  # type: ignore
                k[len("actor_critic.") :]: v
                for k, v in ckpt_dict["state_dict"].items()
            }
        )

        return cls(
            actor_critic,
            config,
            action_space,
            filtered_obs_space,
            filtered_action_space,
            batch_size,
        )


class PickSkillPolicy(NnSkillPolicy):
    def should_terminate(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> torch.Tensor:
        return torch.zeros(masks.shape[0]).to(masks.device)


class NavSkillPolicy(NnSkillPolicy):
    def should_terminate(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> torch.Tensor:
        if self._config.ORACLE_STOP:
            dist_to_nav_goal = observations[DistToNavGoalSensor.cls_uuid]
            should_stop = dist_to_nav_goal < self._config.ORACLE_STOP_DIST
            return should_stop.float()
        return torch.zeros(masks.shape[0]).to(masks.device)


class OracleNavPolicy(NnSkillPolicy):
    def should_terminate(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> torch.Tensor:
        return torch.zeros(masks.shape[0]).to(masks.device)

    @classmethod
    def from_config(cls, config, observation_space, action_space, batch_size):
        return cls(
            None,
            config,
            action_space,
            observation_space,
            action_space,
            batch_size,
        )
