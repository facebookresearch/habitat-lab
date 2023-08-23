import torch

from habitat_baselines.rl.hrl.skills.skill import SkillPolicy
from habitat_baselines.rl.hrl.utils import find_action_range
from habitat_baselines.rl.ppo.policy import PolicyActionData


class MoveSkillPolicy(SkillPolicy):
    def __init__(
        self,
        config,
        action_space,
        batch_size,
    ):
        super().__init__(config, action_space, batch_size, True)
        self._turn_power_fwd = config.turn_power_x
        self._turn_power_side = config.turn_power_y
        self._nav_ac_start, _ = find_action_range(
            action_space, "base_velocity"
        )

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        full_action,
        deterministic=False,
    ):
        full_action = torch.zeros(
            (masks.shape[0], self._full_ac_size), device=masks.device
        )
        if self._turn_power_fwd != 0:
            full_action[:, self._nav_ac_start] = self._turn_power_fwd
        if self._turn_power_side != 0:
            full_action[:, self._nav_ac_start + 1] = self._turn_power_side
        return PolicyActionData(
            actions=full_action, rnn_hidden_states=rnn_hidden_states
        )
