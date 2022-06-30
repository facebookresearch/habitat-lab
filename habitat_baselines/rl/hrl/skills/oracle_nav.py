from dataclasses import dataclass

import magnum as mn
import numpy as np
import torch

from habitat.core.spaces import ActionSpace
from habitat.tasks.rearrange.rearrange_sensors import (
    AbsGoalSensor,
    AbsTargetStartSensor,
    LocalizationSensor,
)
from habitat.tasks.rearrange.sub_tasks.nav_to_obj_sensors import (
    NavGoalSensor,
    OracleNavigationActionSensor,
)
from habitat.tasks.utils import get_angle
from habitat_baselines.common.logging import baselines_logger
from habitat_baselines.rl.hrl.skills.nn_skill import NnSkillPolicy


class OracleNavPolicy(NnSkillPolicy):
    @dataclass(frozen=True)
    class OracleNavArgs:
        obj_idx: int
        is_target: bool

    _is_at_targ: torch.BoolTensor

    def __init__(
        self,
        wrap_policy,
        config,
        action_space,
        filtered_obs_space,
        filtered_action_space,
        batch_size,
    ):
        super().__init__(
            wrap_policy,
            config,
            action_space,
            filtered_obs_space,
            filtered_action_space,
            batch_size,
        )
        self._nav_targs = [None for _ in range(batch_size)]
        self._is_at_targ = torch.zeros(batch_size, dtype=torch.bool)

    def to(self, device):
        self._is_at_targ = self._is_at_targ.to(device)
        self._cur_skill_step = self._cur_skill_step.to(device)
        return self

    def _get_multi_sensor_index(self, batch_idx: int, sensor_name: str) -> int:
        return self._cur_skill_args[batch_idx].obj_idx

    def on_enter(
        self,
        skill_arg,
        batch_idx,
        observations,
        rnn_hidden_states,
        prev_actions,
    ):
        ret = super().on_enter(
            skill_arg, batch_idx, observations, rnn_hidden_states, prev_actions
        )
        self._is_at_targ[batch_idx] = False
        self._nav_targs[batch_idx] = observations[NavGoalSensor.cls_uuid][
            batch_idx
        ]
        self._internal_log(
            f"Got nav target {self._nav_targs} on enter", observations
        )
        return ret

    @classmethod
    def from_config(cls, config, observation_space, action_space, batch_size):
        filtered_action_space = ActionSpace(
            {config.NAV_ACTION_NAME: action_space[config.NAV_ACTION_NAME]}
        )
        baselines_logger.debug(
            f"Loaded action space {filtered_action_space} for skill {config.skill_name}"
        )
        return cls(
            None,
            config,
            action_space,
            observation_space,
            filtered_action_space,
            batch_size,
        )

    def _is_skill_done(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> torch.BoolTensor:
        return self._is_at_targ

    def _compute_forward(self, localization):
        # Compute forward direction
        forward = np.array([1.0, 0, 0])
        heading_angle = localization[-1]
        rot_mat = mn.Matrix4.rotation(
            mn.Rad(heading_angle), mn.Vector3(0, 1, 0)
        )
        robot_forward = np.array(rot_mat.transform_vector(forward))
        return robot_forward

    def _parse_skill_arg(self, skill_arg):
        targ_name, targ_idx = skill_arg[-1].split("|")
        return OracleNavPolicy.OracleNavArgs(
            obj_idx=int(targ_idx), is_target=targ_name.startswith("TARGET")
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
        observations = self._select_obs(observations, cur_batch_idx)

        # The oracle nav target should automatically update based on what part
        # of the task we are on.
        batch_nav_targ = observations[OracleNavigationActionSensor.cls_uuid]
        batch_localization = observations[LocalizationSensor.cls_uuid]

        if self._cur_skill_args[cur_batch_idx].is_target:
            batch_obj_targ_pos = observations[AbsGoalSensor.cls_uuid]
        else:
            batch_obj_targ_pos = observations[AbsTargetStartSensor.cls_uuid]

        full_action = torch.zeros(prev_actions.shape, device=masks.device)
        full_action = self._keep_holding_state(full_action, observations)

        for i, (
            nav_targ,
            localization,
            obj_targ_pos,
            final_nav_goal,
        ) in enumerate(
            zip(
                batch_nav_targ,
                batch_localization,
                batch_obj_targ_pos,
                self._nav_targs,
            )
        ):
            if (
                final_nav_goal.sum() == 0
                and observations[NavGoalSensor.cls_uuid][i].sum() != 0
            ):
                # All zeros is a stable nav goal sensor. Update it to recent.
                self._nav_targs[i] = observations[NavGoalSensor.cls_uuid][i]
                final_nav_goal = self._nav_targs[i]
                self._internal_log(
                    f"Updated nav target {i} to {self._nav_targs}",
                    observations,
                )
            robot_pos = localization[:3]

            robot_forward = self._compute_forward(localization)

            # Compute relative target.
            rel_targ = nav_targ - robot_pos

            # Compute heading angle (2D calculation)
            robot_forward = robot_forward[[0, 2]]
            rel_targ = rel_targ[[0, 2]].cpu().numpy()
            rel_pos = (obj_targ_pos - robot_pos)[[0, 2]].cpu().numpy()

            dist_to_final_nav_targ = torch.linalg.norm(
                (final_nav_goal - robot_pos)[[0, 2]]
            ).item()

            rel_angle = get_angle(robot_forward, rel_targ)
            rel_obj_angle = get_angle(robot_forward, rel_pos)

            vel = [0, 0]
            turn_vel = self._config.TURN_VELOCITY
            for_vel = self._config.FORWARD_VELOCITY

            def compute_turn(rel_a, rel):
                is_left = np.cross(robot_forward, rel) > 0
                if is_left:
                    vel = [0, -turn_vel]
                else:
                    vel = [0, turn_vel]
                return vel

            if dist_to_final_nav_targ < self._config.DIST_THRESH:
                # Look at the object
                vel = compute_turn(rel_obj_angle, rel_pos)
            elif rel_angle < self._config.TURN_THRESH:
                # Move towards the target
                vel = [for_vel, 0]
            else:
                # Look at the target waypoint.
                vel = compute_turn(rel_angle, rel_targ)

            if (
                dist_to_final_nav_targ < self._config.DIST_THRESH
                and rel_obj_angle < self._config.LOOK_AT_OBJ_THRESH
            ):
                self._is_at_targ[i] = True

            full_action[
                i, self._ac_start : self._ac_start + self._ac_len
            ] = torch.tensor(vel).to(masks.device)

        return full_action, rnn_hidden_states
