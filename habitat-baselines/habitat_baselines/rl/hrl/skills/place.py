# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import numpy as np
import torch

from habitat.tasks.rearrange.rearrange_sensors import IsHoldingSensor
from habitat_baselines.rl.hrl.skills.pick import PickSkillPolicy


class PlaceSkillPolicy(PickSkillPolicy):
    @dataclass(frozen=True)
    class PlaceSkillArgs:
        obj: int
        targ: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ee_to_target_threshold = 0.1

    def _get_multi_sensor_index(self, batch_idx):
        return [self._cur_skill_args[i].targ for i in batch_idx]

    def _mask_pick(self, action, observations):
        # Mask out the grasp if the object is already released.
        is_not_holding = 1 - observations[IsHoldingSensor.cls_uuid].view(-1)
        for i in torch.nonzero(is_not_holding):
            # Do not regrasp the object once it is released.
            action.actions[i, self._grip_ac_idx] = -1.0
        return action

    def _force_pick(self, action, observations):
        # Mask out the grasp if the object is already released.
        for i in range(action.actions.shape[0]):
            # Do not regrasp the object once it is released.
            action.actions[i, self._grip_ac_idx] = 1.0
        return action

    def _is_skill_done(
        self,
        observations,
    ) -> torch.BoolTensor:
        # Is the agent not holding an object and is the end-effector at the
        # resting position?
        is_holding = (
            observations[IsHoldingSensor.cls_uuid].view(-1).type(torch.bool)
        )
        current_joint_pos = observations["joint"].cpu().numpy()
        is_reset_done = (
            torch.as_tensor(
                np.abs(current_joint_pos - self._rest_state).max(-1),
                dtype=torch.float32,
            )
            < self._arm_retract_success_threshold
        )
        is_reset_done = is_reset_done.to(is_holding.device)

        return (~is_holding * is_reset_done).type(torch.bool)

    def _parse_skill_arg(self, skill_arg):
        obj = int(skill_arg[0].split("|")[1])
        targ = int(skill_arg[1].split("|")[1])
        return PlaceSkillPolicy.PlaceSkillArgs(obj=obj, targ=targ)

    def _can_drop(self, observations):
        """Check if Spot can drop the object or not by checking the distance between the gripper to the target"""
        ee_to_target = observations["obj_start_sensor"]
        return torch.norm(ee_to_target) < self._ee_to_target_threshold

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        # Store is_holding flag
        is_holding = observations[IsHoldingSensor.cls_uuid].view(-1)

        # Overwrite the observation to let skill know now it is in the grasping step
        if self._config.use_pick_skill_as_place_skill:
            observations[IsHoldingSensor.cls_uuid] = torch.zeros(
                observations[IsHoldingSensor.cls_uuid].shape
            )

        action = super()._internal_act(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            cur_batch_idx,
            deterministic,
        )

        if self._config.use_pick_skill_as_place_skill and is_holding:
            # Keep holding the object until it can drop
            action = self._force_pick(action, observations)
        else:
            action = self._mask_pick(action, observations)

        if not is_holding and self._need_reset_arm:
            action = self._retract_arm_action(observations, action)

        if (
            self._can_drop(observations)
            and self._config.use_pick_skill_as_place_skill
        ):
            # Force the robot to drop the object
            action.actions[0, self._grip_ac_idx] = -1.0

        return action
