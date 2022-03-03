import os.path as osp

import gym.spaces as spaces
import torch
import torch.nn as nn
import yaml

from habitat.core.spaces import ActionSpace
from habitat.tasks.rearrange.multi_task.rearrange_pddl import parse_func
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.utils.common import get_num_actions


class HighLevelPolicy:
    # def __init__(self, config, task_spec_file, num_envs):
    #    self._config = config

    def get_next_skill(
        self, observations, rnn_hidden_states, prev_actions, masks, plan_masks
    ):
        pass


class GtHighLevelPolicy:
    def __init__(self, config, task_spec_file, num_envs, skill_name_to_idx):
        with open(task_spec_file, "r") as f:
            task_spec = yaml.safe_load(f)
        self._solution_actions = [
            parse_func(sol_step) for sol_step in task_spec["solution"]
        ]
        self._next_sol_idxs = torch.zeros(num_envs)
        self._num_envs = num_envs
        self._skill_name_to_idx = skill_name_to_idx

    def get_next_skill(
        self, observations, rnn_hidden_states, prev_actions, masks, plan_masks
    ):
        next_skill = torch.zeros(self._num_envs, device=prev_actions.device)
        skill_args_tensor = torch.zeros(self._num_envs)
        for batch_idx, should_plan in enumerate(plan_masks):
            if should_plan == 1.0:
                skill_name, skill_args = self._solution_actions[
                    self._next_sol_idxs[batch_idx].item()
                ]
                next_skill[batch_idx] = self._skill_name_to_idx[skill_name]
                # Need to convert the name into the corresponding target index.
                # For now use a hack and assume the name correctly indexes
                if len(skill_args) > 0:
                    targ_idx = int(skill_args.split("|")[1])
                    skill_args_tensor[batch_idx] = targ_idx
                self._next_sol_idxs[batch_idx] += 1
        return next_skill, skill_args_tensor


@baseline_registry.register_policy
class HierarchicalPolicy(Policy):
    def __init__(
        self, config, full_config, observation_space, action_space, num_envs
    ):
        super().__init__()

        self._action_space = action_space

        # Maps (skill idx -> skill)
        self._skills = {}
        self._name_to_idx = {}

        for i, (skill_name, skill_config) in enumerate(config.skills.items()):
            cls = eval(skill_config.skill_name)
            skill_policy = cls.from_config(
                skill_config, observation_space, action_space
            )
            self._skills[i] = skill_policy
            self._name_to_idx[skill_name] = i

        self._num_envs = num_envs
        self._call_high_level = torch.ones(self._num_envs)
        self._cur_skills: torch.Tensor = torch.zeros(self._num_envs)

        high_level_cls = eval(config.high_level_policy.name)
        self._high_level_policy: HighLevelPolicy = high_level_cls(
            config.high_level_policy,
            osp.join(
                full_config.TASK_CONFIG.TASK.TASK_SPEC_BASE_PATH,
                full_config.TASK_CONFIG.TASK.TASK_SPEC + ".yaml",
            ),
            num_envs,
            self._name_to_idx,
        )

    def eval(self):
        pass

    @property
    def num_recurrent_layers(self):
        return self._skills[0].num_recurrent_layers

    def parameters(self):
        return self._skills[0].parameters()

    def to(self, device):
        for skill in self._skills.values():
            skill.to(device)
        self._call_high_level = self._call_high_level.to(device)

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):

        batched_observations = {
            k: v.unsqueeze(1) for k, v in observations.items()
        }
        batched_rnn_hidden_states = rnn_hidden_states.unsqueeze(1)
        batched_prev_actions = prev_actions.unsqueeze(1)
        batched_masks = masks.unsqueeze(1)

        # Check if skills should terminate.
        for batch_idx, skill_idx in enumerate(self._cur_skills):
            should_terminate = self._skills[skill_idx.item()].should_terminate(
                TensorDict(
                    {k: v[batch_idx] for k, v in batched_observations.items()}
                ),
                batched_rnn_hidden_states[batch_idx],
                batched_prev_actions[batch_idx],
                batched_masks[batch_idx],
            )
            self._call_high_level[batch_idx] = should_terminate

        # Always call high-level if the episode is over.
        self._call_high_level = torch.clamp(
            self._call_high_level + (~masks).float().view(-1), 0.0, 1.0
        )

        # If any skills want to terminate invoke the high-level policy to get
        # the next skill.
        if self._call_high_level.sum() > 0:
            (
                new_skills,
                new_skill_args,
            ) = self._high_level_policy.get_next_skill(
                observations,
                rnn_hidden_states,
                prev_actions,
                masks,
                self._call_high_level,
            )

            self._cur_skills = self._call_high_level * new_skills

            for new_skill_batch_idx in torch.nonzero(self._call_high_level):
                skill_idx = self._cur_skills[new_skill_batch_idx.item()]
                skill = self._skills[skill_idx.item()]
                skill.on_enter(new_skill_args[new_skill_batch_idx])

        actions = torch.zeros(
            self._num_envs, get_num_actions(self._action_space)
        )
        for batch_idx, skill_idx in enumerate(self._cur_skills):
            action = self._skills[skill_idx.item()].act(
                TensorDict(
                    {k: v[batch_idx] for k, v in batched_observations.items()}
                ),
                batched_rnn_hidden_states[batch_idx],
                batched_prev_actions[batch_idx],
                batched_masks[batch_idx],
            )
            actions[batch_idx] = action

        return None, actions, None, rnn_hidden_states

    @classmethod
    def from_config(cls, config, observation_space, action_space):
        return cls(
            config.RL.POLICY,
            config,
            observation_space,
            action_space,
            config.NUM_ENVIRONMENTS,
        )


class NnSkillPolicy(Policy):
    def __init__(
        self,
        wrap_policy,
        config,
        action_space,
        filtered_obs_space,
        filtered_action_space,
    ):
        self._wrap_policy = wrap_policy
        self._action_space = action_space
        self._config = config
        self._filtered_obs_space = filtered_obs_space
        self._filtered_action_space = filtered_action_space
        self._ac_start = 0
        self._ac_len = get_num_actions(filtered_action_space)
        for k in action_space:
            if k not in filtered_action_space.keys():
                self._ac_start += get_num_actions(action_space[k])
            else:
                break

    def should_terminate(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> torch.Tensor:
        return torch.zeros(observations.shape[0]).to(masks.device)

    def on_enter(self, skill_args):
        pass

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
    def from_config(cls, config, observation_space, action_space):
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
    def from_config(cls, config, observation_space, action_space):
        return cls(
            None,
            config,
            action_space,
            observation_space,
            action_space,
        )


class NavSkillPolicy(NnSkillPolicy):
    def should_terminate(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> torch.Tensor:
        return torch.zeros(masks.shape[0]).to(masks.device)
