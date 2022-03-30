from dataclasses import dataclass
from typing import Any, List, Tuple

import gym.spaces as spaces
import magnum as mn
import numpy as np
import torch
import torch.nn as nn

from habitat.core.spaces import ActionSpace
from habitat.tasks.rearrange.rearrange_sensors import (
    AbsGoalSensor,
    AbsTargetStartSensor,
    IsHoldingSensor,
    LocalizationSensor,
    RelativeRestingPositionSensor,
)
from habitat.tasks.rearrange.sub_tasks.nav_to_obj_sensors import (
    DistToNavGoalSensor,
    NavGoalSensor,
    NavRotToGoalSensor,
    OracleNavigationActionSensor,
)
from habitat.tasks.utils import get_angle
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.logging import logger
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.hrl.high_level_policy import (
    GtHighLevelPolicy,
    HighLevelPolicy,
)
from habitat_baselines.rl.ppo.policy import Policy
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

        self._cur_skill_step = torch.zeros(self._batch_size)

        self._cur_skill_args: List[Any] = [
            None for _ in range(self._batch_size)
        ]

        for k in action_space:
            if k not in filtered_action_space.spaces.keys():
                self._ac_start += get_num_actions(action_space[k])
            else:
                break

        logger.info(
            f"Skill {self._config.skill_name}: action offset {self._ac_start}, action length {self._ac_len}"
        )

    def _internal_log(self, s):
        logger.info(f"Skill {self._config.skill_name}: {s}")

    def _select_obs(self, obs, cur_batch_idx):
        """
        Selects out the part of the observation that corresponds to the current goal of the skill.
        """
        for k in self._config.OBS_SKILL_INPUTS:
            cur_multi_sensor_index = self._get_multi_sensor_index(
                cur_batch_idx, k
            )
            if k not in obs:
                raise ValueError(
                    f"Skill {self._config.skill_name}: Could not find {k} out of {obs.keys()}"
                )
            entity_positions = obs[k].view(1, -1, 3)
            obs[k] = entity_positions[:, cur_multi_sensor_index]
        return obs

    def _get_multi_sensor_index(self, batch_idx: int, sensor_name: str) -> int:
        """
        Gets the index to select the observation object index in `_select_obs`.
        Used when there are multiple possible goals in the scene, such as
        multiple objects to possibly rearrange.
        """
        return self._cur_skill_args[batch_idx]

    def should_terminate(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :returns: A (batch_size,) size tensor where 1 indicates the skill wants to end and 0 if not.
        """
        is_skill_done = self._is_skill_done(
            observations, rnn_hidden_states, prev_actions, masks
        )
        bad_terminate = self._cur_skill_step > self._config.MAX_SKILL_STEPS
        if bad_terminate.sum() > 0:
            self._internal_log(
                f"Bad terminating due to timeout {self._cur_skill_step}, {bad_terminate}"
            )

        return is_skill_done, bad_terminate

    def _is_skill_done(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> torch.Tensor:
        """
        :returns: A (batch_size,) size tensor where 1 indicates the skill wants to end and 0 if not.
        """
        return torch.zeros(observations.shape[0]).to(masks.device)

    def _parse_skill_arg(self, skill_arg: str) -> Any:
        """
        Parses the skill argument string identifier and returns parsed skill argument information.
        """
        return skill_arg

    def on_enter(
        self,
        skill_arg: List[str],
        batch_idx: int,
        observations,
        rnn_hidden_states,
        prev_actions,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Passes in the data at the current `batch_idx`
        :returns: The new hidden state and prev_actions ONLY at the batch_idx.
        """
        self._cur_skill_args[batch_idx] = self._parse_skill_arg(skill_arg)
        self._internal_log(
            f"Entering skill with arguments {skill_arg} parsed to {self._cur_skill_args[batch_idx]}"
        )

        self._cur_skill_step[batch_idx] = 0

        return (
            rnn_hidden_states[batch_idx] * 0.0,
            prev_actions[batch_idx] * 0.0,
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
        if self._wrap_policy is not None:
            self._wrap_policy.to(device)
        self._cur_skill_step = self._cur_skill_step.to(device)

    def _internal_act(
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
                k: observations[k]
                for k in self._filtered_obs_space.spaces.keys()
            }
        )

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

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        self._cur_skill_step[cur_batch_idx] += 1
        return self._internal_act(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            cur_batch_idx,
            deterministic,
        )

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

        expected_obs_keys = list(
            set(
                policy_cfg.RL.POLICY.include_visual_keys
                + policy_cfg.RL.GYM_OBS_KEYS
            )
        )
        filtered_obs_space = spaces.Dict(
            {k: observation_space.spaces[k] for k in expected_obs_keys}
        )

        for k in config.OBS_SKILL_INPUTS:
            space = filtered_obs_space.spaces[k]
            # There is always a 3D position
            filtered_obs_space.spaces[k] = truncate_obs_space(space, 3)
        logger.info(
            f"Skill {config.skill_name}: Loaded observation space {filtered_obs_space}"
        )

        filtered_action_space = ActionSpace(
            {
                k: action_space[k]
                for k in policy_cfg.TASK_CONFIG.TASK.POSSIBLE_ACTIONS
            }
        )
        logger.info(
            f"Loaded action space {filtered_action_space} for skill {config.skill_name}"
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
                f"Could not load checkpoint for skill {config.skill_name}"
            ) from e

        return cls(
            actor_critic,
            config,
            action_space,
            filtered_obs_space,
            filtered_action_space,
            batch_size,
        )


class PickSkillPolicy(NnSkillPolicy):
    def _is_skill_done(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> torch.Tensor:
        # Is the agent holding the object and is the end-effector at the
        # resting position?
        rel_resting_pos = torch.norm(
            observations[RelativeRestingPositionSensor.cls_uuid], dim=-1
        )
        is_within_thresh = rel_resting_pos < self._config.AT_RESTING_THRESHOLD
        is_holding = observations[IsHoldingSensor.cls_uuid].view(-1)
        return is_holding * is_within_thresh.float()

    def _parse_skill_arg(self, skill_arg):
        self._internal_log(f"Parsing skill argument {skill_arg}")
        return int(skill_arg[0].split("|")[1])

    def _mask_pick(self, action, observations):
        is_holding = observations[IsHoldingSensor.cls_uuid].view(-1)
        for i in torch.nonzero(is_holding):
            # Do not release the object once it is held
            action[i, self._ac_start + self._ac_len] = 1.0
        return action

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        action, hxs = super()._internal_act(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            cur_batch_idx,
            deterministic,
        )
        action = self._mask_pick(action, observations)
        # Mask out the release if the object is already held.
        return action, hxs


class PlaceSkillPolicy(NnSkillPolicy):
    @dataclass(frozen=True)
    class PlaceSkillArgs:
        obj: int
        targ: int

    def _get_multi_sensor_index(self, batch_idx: int, sensor_name: str) -> int:
        return self._cur_skill_args[batch_idx].targ

    def _is_skill_done(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> torch.Tensor:
        # Is the agent not holding an object and is the end-effector at the
        # resting position?
        rel_resting_pos = torch.norm(
            observations[RelativeRestingPositionSensor.cls_uuid], dim=-1
        )
        is_within_thresh = rel_resting_pos < self._config.AT_RESTING_THRESHOLD
        is_not_holding = 1 - observations[IsHoldingSensor.cls_uuid].view(-1)
        is_done = is_not_holding * is_within_thresh.float()
        if is_done.sum() > 0:
            self._internal_log(
                f"Terminating with {rel_resting_pos} and {is_not_holding}"
            )
        return is_done

    def _parse_skill_arg(self, skill_arg):
        obj = int(skill_arg[0].split("|")[1])
        targ = int(skill_arg[1].split("|")[1])
        return PlaceSkillPolicy.PlaceSkillArgs(obj=obj, targ=targ)


class NavSkillPolicy(NnSkillPolicy):
    def _is_skill_done(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> torch.Tensor:
        if self._config.ORACLE_STOP:
            dist_to_nav_goal = observations[DistToNavGoalSensor.cls_uuid]
            rot_to_nav_goal = observations[NavRotToGoalSensor.cls_uuid]
            should_stop = (
                dist_to_nav_goal < self._config.ORACLE_STOP_DIST
            ) * (rot_to_nav_goal < self._config.ORACLE_STOP_ANGLE_DIST)
            return should_stop.float()
        return torch.zeros(masks.shape[0]).to(masks.device)

    def _parse_skill_arg(self, skill_arg):
        return int(skill_arg[-1].split("|")[1])


class OracleNavPolicy(NnSkillPolicy):
    @dataclass(frozen=True)
    class OracleNavArgs:
        obj_idx: int
        is_target: bool

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
        self._is_at_targ = torch.zeros(batch_size)
        self._nav_targs = torch.zeros(batch_size, 3)

    def to(self, device):
        self._is_at_targ = self._is_at_targ.to(device)
        self._nav_targs = self._nav_targs.to(device)
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
        self._is_at_targ[batch_idx] = 0.0
        self._nav_targs[batch_idx] = observations[NavGoalSensor.cls_uuid][
            batch_idx
        ]
        return super().on_enter(
            skill_arg, batch_idx, observations, rnn_hidden_states, prev_actions
        )

    @classmethod
    def from_config(cls, config, observation_space, action_space, batch_size):
        filtered_action_space = ActionSpace(
            {config.NAV_ACTION_NAME: action_space[config.NAV_ACTION_NAME]}
        )
        logger.info(
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
    ) -> torch.Tensor:
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
                # Look at the target.
                vel = compute_turn(rel_angle, rel_targ)

            if (
                dist_to_final_nav_targ < self._config.DIST_THRESH
                and rel_obj_angle < self._config.LOOK_AT_OBJ_THRESH
            ):
                self._is_at_targ[i] = 1.0

            full_action[
                i, self._ac_start : self._ac_start + self._ac_len
            ] = torch.tensor(vel).to(masks.device)

        return full_action, rnn_hidden_states
