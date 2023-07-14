from habitat_baselines.rl.hrl.skills.skill import SkillPolicy
from habitat_baselines.rl.hrl.utils import find_action_range


class TurnSkillPolicy(SkillPolicy):
    def __init__(
        self,
        config,
        action_space,
        batch_size,
    ):
        super().__init__(config, action_space, batch_size, True)
        self._turn_power = config.turn_power
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
        full_action[:, self._nav_ac_start + 1] = self._turn_power
        return full_action, rnn_hidden_states
