import torch
import yaml

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo.policy import Policy


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
        self._solution_actions = task_spec["solution"]
        self._next_sol_idxs = torch.zeros(num_envs)
        self._num_envs = num_envs
        self._skill_name_to_idx = skill_name_to_idx

    def get_next_skill(
        self, observations, rnn_hidden_states, prev_actions, masks, plan_masks
    ):
        next_skill = torch.zeros(self._num_envs)
        for batch_idx, should_plan in enumerate(plan_masks):
            if should_plan == 1.0:
                skill_name = self._solution_actions[batch_idx]
                next_skill[batch_idx] = self._skill_name_to_idx[skill_name]
        return next_skill


@baseline_registry.register_policy
class HierarchicalPolicy(Policy):
    def __init__(
        self, config, full_config, observation_space, action_space, num_envs
    ):
        super().__init__()

        # Maps (skill idx -> skill)
        self._action_space = action_space
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
        # Shape [batch_size,]
        self._cur_skills: torch.Tensor = torch.zeros(self._num_envs)

        high_level_cls = eval(config.high_level_policy.name)
        self._high_level_policy: HighLevelPolicy = high_level_cls(
            config.high_level_policy,
            config.TASK_CONFIG.TASK.TASK_SPEC_BASE_PATH,
            num_envs,
            self._name_to_idx,
        )

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):

        batched_observations = observations.unsqueeze(1)
        batched_rnn_hidden_states = rnn_hidden_states.unsqueeze(1)
        batched_prev_actions = prev_actions.unsqueeze(1)
        batched_masks = masks.unsqueeze(1)

        for batch_idx, skill_idx in enumerate(self._cur_skills):
            should_terminate = self._skills[skill_idx].should_terminate(
                batched_observations[batch_idx],
                batched_rnn_hidden_states[batch_idx],
                batched_prev_actions[batch_idx],
                batched_masks[batch_idx],
            )
            self._call_high_level[batch_idx] = should_terminate

        # Always call high-level if the episode is over.
        self._call_high_level *= 1 - masks

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
                skill_idx = self._cur_skills[new_skill_batch_idx]
                skill = self._skills[skill_idx]
                skill.on_enter(new_skill_args[new_skill_batch_idx])

        actions = torch.zeros(self._num_envs, *self._action_space.shape)
        for batch_idx, skill_idx in enumerate(self._cur_skills):
            action = self._skills[skill_idx].act(
                batched_observations[batch_idx],
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
    def __init__(self, wrap_policy, config, action_space):
        self._wrap_policy = wrap_policy
        self._action_space = action_space
        self._config = config

    def should_terminate(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> torch.Tensor:
        return torch.zeros(observations.shape[0])

    def on_enter(self, skill_args):
        pass

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        return self._wrap_policy.act(
            observations, rnn_hidden_states, prev_actions, masks, deterministic
        )

    @classmethod
    def from_config(cls, config, observation_space, action_space):
        # Load the wrap policy from file
        if len(config.LOAD_CKPT_FILE) == 0:
            raise ValueError("Need to specify load location")

        ckpt_dict = torch.load(config.LOAD_CKPT_FILE, map_location="cpu")
        policy = baseline_registry.get_policy(config.name)
        actor_critic = policy.from_config(
            ckpt_dict["config"], observation_space, action_space
        )
        actor_critic.load_state_dict(
            {  # type: ignore
                k[len("actor_critic.") :]: v
                for k, v in ckpt_dict["state_dict"].items()
            }
        )

        return cls(actor_critic, config, action_space)


class PickSkillPolicy(NnSkillPolicy):
    def should_terminate(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> torch.Tensor:
        return torch.zeros(observations.shape[0])


class NavSkillPolicy(NnSkillPolicy):
    def should_terminate(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> torch.Tensor:
        return torch.zeros(observations.shape[0])
