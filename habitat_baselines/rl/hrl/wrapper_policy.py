from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo.policy import Policy
import torch
from habitat_baselines.utils.common import get_num_actions


@baseline_registry.register_policy
class WrapperPolicy(Policy):
    def __init__(self, wrapped_policy, policy_cfg, n_envs, action_dim):
        super().__init__()
        self._wrapped_policy = wrapped_policy
        self._timesteps = torch.zeros(n_envs)
        self._n_envs = n_envs
        self._policy_cfg = policy_cfg
        self._active_envs = list(range(self._n_envs))

    def add_paused_envs(self, paused):
        for i in reversed(paused):
            self._active_envs.pop(i)

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        if masks.shape[0] == self._timesteps.shape[0]:
            self._timesteps[self._active_envs] *= masks.view(-1).cpu()

        should_act = (
            self._timesteps[self._active_envs] % self._policy_cfg.act_freq
        ) == 0

        _, actions, _, new_rnn_hxs = self._wrapped_policy.act(
            observations, rnn_hidden_states, prev_actions, masks, deterministic
        )

        for i, did_act in enumerate(should_act):
            if not did_act:
                new_rnn_hxs[i] = rnn_hidden_states[i]
                actions[i, 1:] *= 0.0

        self._timesteps += 1
        # When we are at the goal, drop the object.
        is_holding = observations["is_holding"].bool().view(-1)
        should_release = (
            torch.linalg.norm(observations["obj_goal_sensor"], dim=-1)
            < self._policy_cfg.release_thresh
        )
        n_envs = should_release.size(0)
        for env_i in range(n_envs):
            if is_holding[env_i] and not should_release[env_i]:
                # Keep holding
                actions[:, :1] = 1.0

        return (None, actions, None, new_rnn_hxs, should_act)

    @property
    def num_recurrent_layers(self) -> int:
        return self._wrapped_policy.num_recurrent_layers

    def parameters(self):
        return self._wrapped_policy.parameters()

    def eval(self):
        return self._wrapped_policy.eval()

    def state_dict(self):
        return self._wrapped_policy.state_dict()

    @property
    def critic(self):
        return self._wrapped_policy.critic

    def to(self, device):
        self._wrapped_policy.to(device)

    @classmethod
    def from_config(cls, config, observation_space, action_space):
        policy = baseline_registry.get_policy(config.RL.POLICY.wrapped_name)
        actor_critic = policy.from_config(
            config, observation_space, action_space
        )

        return WrapperPolicy(
            actor_critic,
            config.RL.POLICY,
            config.NUM_ENVIRONMENTS,
            get_num_actions(action_space),
        )
