# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from habitat.core.simulator import Observations
from habitat.tasks.rearrange.rearrange_sensors import (
    IsHoldingSensor,
    RelativeRestingPositionSensor,
)
from habitat_baselines.rl.hrl.skills.nn_skill import NnSkillPolicy
from habitat_baselines.rl.hrl.utils import SkillIOManager, find_action_range
from habitat_baselines.rl.ppo.policy import PolicyActionData
from habitat_baselines.utils.common import get_num_actions


class PickSkillPolicy(NnSkillPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Get the action space
        action_space = args[2]
        batch_size = args[-1]
        for k, _ in action_space.items():
            if k == "pick_base_velocity":
                self.pick_start_id, self.pick_len = find_action_range(
                    action_space, "pick_base_velocity"
                )
            if k == "base_velocity":
                self.org_pick_start_id, self.org_pick_len = find_action_range(
                    action_space, "base_velocity"
                )
            if k == "arm_action":
                self.arm_start_id, self.arm_len = find_action_range(
                    action_space, "arm_action"
                )
            if k == "reset_arm_action":
                (
                    self.reset_arm_start_id,
                    self.reset_arm_len,
                ) = find_action_range(action_space, "reset_arm_action")

        # Get the skill io manager
        config = args[1]
        self.sm = SkillIOManager()
        self._num_ac = get_num_actions(action_space)
        self._rest_state = np.array(config.reset_joint_state)
        self._need_reset_arm = True
        self._dist_to_rest_state = 0.05

        self.sm.init_hidden_state(
            batch_size,
            self._wrap_policy.net._hidden_size,
            self._wrap_policy.num_recurrent_layers,
        )
        self.sm.init_prev_action(batch_size, self._num_ac)
        self._step = 0

    def _is_skill_done(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        batch_idx,
    ) -> torch.BoolTensor:
        # Is the agent holding the object and is the end-effector at the
        # resting position?
        rel_resting_pos = torch.norm(
            observations[RelativeRestingPositionSensor.cls_uuid], dim=-1
        )
        is_within_thresh = rel_resting_pos < self._config.at_resting_threshold
        is_holding = observations[IsHoldingSensor.cls_uuid].view(-1)

        current_joint_pos = observations["joint"].cpu().numpy()
        is_reset_down = (
            torch.as_tensor(
                np.abs(current_joint_pos - self._rest_state).max(-1),
                dtype=torch.float32,
            )
            < self._dist_to_rest_state
        )
        is_reset_down = is_reset_down.to(is_holding.device)
        is_done = (is_holding * is_within_thresh * is_reset_down).type(
            torch.bool
        )
        if is_done.sum() > 0:
            self.sm.hidden_state[batch_idx][is_done] *= 0
            self.sm.prev_action[batch_idx][is_done] *= 0

        return is_done

    def _parse_skill_arg(self, skill_arg):
        self._internal_log(f"Parsing skill argument {skill_arg}")
        return int(skill_arg[0].split("|")[1])

    def _mask_pick(
        self, action: PolicyActionData, observations
    ) -> PolicyActionData:
        # Mask out the release if the object is already held.
        is_holding = observations[IsHoldingSensor.cls_uuid].view(-1)
        for i in torch.nonzero(is_holding):
            # Do not release the object once it is held
            action.actions[i, self._grip_ac_idx] = 1.0
        return action

    def should_terminate(
        self,
        observations: Observations,
        rnn_hidden_states: torch.Tensor,
        prev_actions: torch.Tensor,
        masks: torch.Tensor,
        actions: torch.Tensor,
        hl_wants_skill_term: torch.BoolTensor,
        batch_idx: List[int],
        skill_name: List[str],
        log_info: List[Dict[str, Any]],
    ) -> Tuple[torch.BoolTensor, torch.BoolTensor, torch.Tensor]:
        arm_slice = slice(
            self.arm_start_id, self.arm_start_id + self.arm_len - 1
        )
        prev_arm_action = actions[:, arm_slice]

        is_skill_done, bad_terminate, actions = super().should_terminate(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            actions,
            hl_wants_skill_term,
            batch_idx,
            skill_name,
            log_info,
        )

        rest_state = torch.from_numpy(self._rest_state).to(
            device=actions.device, dtype=actions.dtype
        )
        reset_arm_slice = slice(
            self.reset_arm_start_id,
            self.reset_arm_start_id + self.reset_arm_len,
        )

        self._step += 1
        # We use is_skill_done since there are two cases when we do termination:
        # (1) the high-level policy wants to do so, and
        # (2) exceeding the max steps
        # (3) there is no a single the target object that is within graspable distance

        # filter_observations = self._select_obs(self._get_filtered_obs(observations, batch_idx), batch_idx)
        # try:
        #     obj_dis_sensor = filter_observations["obj_start_sensor"]
        # except Exception:
        #    obj_dis_sensor = filter_observations["obj_goal_sensor"]
        # obj_dis_sensor = torch.unsqueeze(obj_dis_sensor, 0)
        # obj_dis_sensor = torch.reshape(obj_dis_sensor, (-1, 2, 3))

        # Within certain meters of grasping point
        # cannot_grasp = torch.linalg.norm(obj_dis_sensor, dim=2, ord=2) > 2
        # cannot_grasp = torch.linalg.norm(obj_dis_sensor, dim=1, ord=2) > 4
        # cannot_grasp = torch.sum(cannot_grasp, dim=-1) == 2
        # cannot_grasp = cannot_grasp.to(is_skill_done.device)

        # Update the is_skill_done
        # is_skill_done = is_skill_done | cannot_grasp

        if is_skill_done.sum() > 0:
            current_joint_pos = observations["joint"][is_skill_done, ...]
            delta = rest_state - current_joint_pos
            actions[is_skill_done, reset_arm_slice] = delta
            # We mutiply by -1 to eliminate this action
            actions[is_skill_done, arm_slice] = -prev_arm_action[is_skill_done]

        return is_skill_done, bad_terminate, actions

    def to(self, device):
        super().to(device)
        self.sm.to(device)

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        action = super()._internal_act(
            observations,
            self.sm.hidden_state[cur_batch_idx],
            self.sm.prev_action[cur_batch_idx],
            masks,
            cur_batch_idx,
            deterministic,
        )

        # The skill outputs a base velocity, but we want to
        # set it to pick_base_velocity. This is because we want
        # to potentially have different lin_speeds
        # For those velocities
        base_vel_index = slice(
            self.org_pick_start_id, self.org_pick_start_id + self.org_pick_len
        )
        ac_vel_index = slice(
            self.pick_start_id, self.pick_start_id + self.pick_len
        )
        arm_slice = slice(
            self.arm_start_id, self.arm_start_id + self.arm_len - 1
        )

        action.actions[:, ac_vel_index] = action.actions[:, base_vel_index]
        size = action.actions[:, base_vel_index].shape
        action.actions[:, base_vel_index] = torch.zeros(size)
        action = self._mask_pick(action, observations)

        # Update the hidden state / action
        self.sm.hidden_state[cur_batch_idx] = action.rnn_hidden_states
        self.sm.prev_action[cur_batch_idx] = action.actions

        is_holding = observations[IsHoldingSensor.cls_uuid].view(-1)

        rest_state = torch.from_numpy(self._rest_state).to(
            device=action.actions.device, dtype=action.actions.dtype
        )
        # Do not release the object once it is held
        if self._need_reset_arm and is_holding.sum() > 0:
            current_joint_pos = observations["joint"][is_holding.bool()]
            delta = rest_state - current_joint_pos
            action.actions[is_holding.bool(), arm_slice] = delta
        return action
