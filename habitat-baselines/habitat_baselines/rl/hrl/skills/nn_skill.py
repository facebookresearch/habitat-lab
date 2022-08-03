import gym.spaces as spaces
import numpy as np
import torch

from habitat.config import Config as CN
from habitat.core.spaces import ActionSpace
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.logging import baselines_logger
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.hrl.skills.skill import SkillPolicy
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

        for k, space in action_space.items():
            if k not in filtered_action_space.spaces.keys():
                self._ac_start += get_num_actions(space)
            else:
                break

        self._internal_log(
            f"Skill {self._config.skill_name}: action offset {self._ac_start}, action length {self._ac_len}"
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
        if self._wrap_policy is not None:
            self._wrap_policy.to(device)

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
    ):
        filtered_obs = self._get_filtered_obs(observations, cur_batch_idx)

        filtered_prev_actions = prev_actions[
            :, self._ac_start : self._ac_start + self._ac_len
        ]
        filtered_obs = self._select_obs(filtered_obs, cur_batch_idx)

        _, action, _, rnn_hidden_states = self._wrap_policy.act(
            filtered_obs,
            rnn_hidden_states,
            filtered_prev_actions,
            masks,
            deterministic,
        )
        full_action = torch.zeros(prev_actions.shape)
        full_action[:, self._ac_start : self._ac_start + self._ac_len] = action
        return full_action, rnn_hidden_states

    @classmethod
    def from_config(cls, config, observation_space, action_space, batch_size):
        # Load the wrap policy from file
        if len(config.LOAD_CKPT_FILE) == 0:
            raise ValueError(
                f"Skill {config.skill_name}: Need to specify LOAD_CKPT_FILE"
            )

        ckpt_dict = torch.load(config.LOAD_CKPT_FILE, map_location="cpu")
        policy = baseline_registry.get_policy(config.name)
        policy_cfg = ckpt_dict["config"]

        if "GYM" not in policy_cfg.TASK_CONFIG:
            # Support loading legacy policies
            # TODO: Remove this eventually and drop support for policies
            # trained on older version of codebase.
            policy_cfg.defrost()
            policy_cfg.TASK_CONFIG.GYM = CN()
            policy_cfg.TASK_CONFIG.GYM.OBS_KEYS = list(
                set(
                    policy_cfg.RL.POLICY.include_visual_keys
                    + policy_cfg.RL.GYM_OBS_KEYS
                )
            )
            policy_cfg.freeze()

        expected_obs_keys = policy_cfg.TASK_CONFIG.GYM.OBS_KEYS
        filtered_obs_space = spaces.Dict(
            {k: observation_space.spaces[k] for k in expected_obs_keys}
        )

        for k in config.OBS_SKILL_INPUTS:
            space = filtered_obs_space.spaces[k]
            # There is always a 3D position
            filtered_obs_space.spaces[k] = truncate_obs_space(space, 3)
        baselines_logger.debug(
            f"Skill {config.skill_name}: Loaded observation space {filtered_obs_space}",
        )

        filtered_action_space = ActionSpace(
            {
                k: action_space[k]
                for k in policy_cfg.TASK_CONFIG.TASK.POSSIBLE_ACTIONS
            }
        )

        if "ARM_ACTION" in filtered_action_space.spaces and (
            policy_cfg.TASK_CONFIG.TASK.ACTIONS.ARM_ACTION.GRIP_CONTROLLER
            is None
        ):
            filtered_action_space["ARM_ACTION"] = spaces.Dict(
                {
                    k: v
                    for k, v in filtered_action_space["ARM_ACTION"].items()
                    if k != "grip_action"
                }
            )

        baselines_logger.debug(
            f"Loaded action space {filtered_action_space} for skill {config.skill_name}",
        )

        actor_critic = policy.from_config(
            policy_cfg, filtered_obs_space, filtered_action_space
        )

        try:
            actor_critic.load_state_dict(
                {  # type: ignore
                    k[len("actor_critic.") :]: v
                    for k, v in ckpt_dict["state_dict"].items()
                }
            )

        except Exception as e:
            raise ValueError(
                f"Could not load checkpoint for skill {config.skill_name} from {config.LOAD_CKPT_FILE}"
            ) from e

        return cls(
            actor_critic,
            config,
            action_space,
            filtered_obs_space,
            filtered_action_space,
            batch_size,
        )
