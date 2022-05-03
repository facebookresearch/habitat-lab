from dataclasses import dataclass
from typing import Any, List, Tuple

import gym.spaces as spaces
import magnum as mn
import numpy as np
import torch

from habitat.core.spaces import ActionSpace
from habitat.tasks.rearrange.rearrange_sensors import (
    AbsGoalSensor,
    AbsTargetStartSensor,
    IsHoldingSensor,
    LocalizationSensor,
    RelativeRestingPositionSensor,
)
from habitat.tasks.rearrange.sub_tasks.nav_to_obj_sensors import (
    NavGoalSensor,
    OracleNavigationActionSensor,
)
from habitat.tasks.utils import get_angle
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.logging import baselines_logger
from habitat_baselines.common.tensor_dict import TensorDict
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


class SkillPolicy(Policy):
    def __init__(
        self,
        config,
        action_space: spaces.Space,
        batch_size,
        should_keep_hold_state: bool = False,
    ):
        """
        :param action_space: The overall action space of the entire task, not task specific.
        """
        self._action_space = action_space
        self._config = config
        self._batch_size = batch_size

        self._cur_skill_step = torch.zeros(self._batch_size)
        self._should_keep_hold_state = should_keep_hold_state

        self._cur_skill_args: List[Any] = [
            None for _ in range(self._batch_size)
        ]

        self._grip_ac_idx = 0
        found_grip = False
        for k, space in action_space.items():
            if k != "ARM_ACTION":
                self._grip_ac_idx += get_num_actions(space)
            else:
                # The last actioin in the arm action is the grip action.
                self._grip_ac_idx += get_num_actions(space) - 1
                found_grip = True
                break
        if not found_grip:
            raise ValueError(f"Could not find grip action in {action_space}")

    def _internal_log(self, s, observations=None):
        baselines_logger.debug(
            f"Skill {self._config.skill_name} @ step {self._cur_skill_step}: {s}"
        )

    def _get_multi_sensor_index(self, batch_idx: int, sensor_name: str) -> int:
        """
        Gets the index to select the observation object index in `_select_obs`.
        Used when there are multiple possible goals in the scene, such as
        multiple objects to possibly rearrange.
        """
        return self._cur_skill_args[batch_idx]

    def _keep_holding_state(
        self, full_action: torch.Tensor, observations
    ) -> torch.Tensor:
        """
        Makes the action so it does not result in dropping or picking up an
        object. Used in navigation and other skills which are not supposed to
        interact through the gripper.
        """
        # Keep the same grip state as the previous action.
        is_holding = observations[IsHoldingSensor.cls_uuid].view(-1)
        # If it is not holding (0) want to keep releasing -> output -1.
        # If it is holding (1) want to keep grasping -> output +1.
        full_action[:, self._grip_ac_idx] = is_holding + (is_holding - 1.0)
        return full_action

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
        if is_skill_done:
            self._internal_log(
                "Requested skill termination",
                observations,
            )

        if self._config.MAX_SKILL_STEPS > 0:
            bad_terminate = self._cur_skill_step > self._config.MAX_SKILL_STEPS
        else:
            bad_terminate = torch.zeros(
                self._cur_skill_step.shape, device=self._cur_skill_step.device
            )
        if bad_terminate.sum() > 0:
            self._internal_log(
                f"Bad terminating due to timeout {self._cur_skill_step}, {bad_terminate}",
                observations,
            )

        return is_skill_done, bad_terminate

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
        self._cur_skill_step[batch_idx] = 0
        self._cur_skill_args[batch_idx] = self._parse_skill_arg(skill_arg)

        self._internal_log(
            f"Entering skill with arguments {skill_arg} parsed to {self._cur_skill_args[batch_idx]}",
            observations,
        )

        return (
            rnn_hidden_states[batch_idx] * 0.0,
            prev_actions[batch_idx] * 0.0,
        )

    @classmethod
    def from_config(cls, config, observation_space, action_space, batch_size):
        return cls(config, action_space, batch_size)

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        """
        :returns: Predicted action and next rnn hidden state.
        """
        self._cur_skill_step[cur_batch_idx] += 1
        action, hxs = self._internal_act(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            cur_batch_idx,
            deterministic,
        )

        if self._should_keep_hold_state:
            action = self._keep_holding_state(action, observations)
        return action, hxs

    def to(self, device):
        self._cur_skill_step = self._cur_skill_step.to(device)

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

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()


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


class WaitSkillPolicy(SkillPolicy):
    def __init__(
        self,
        config,
        action_space: spaces.Space,
        batch_size,
    ):
        super().__init__(config, action_space, batch_size, True)
        self._wait_time = -1

    def _parse_skill_arg(self, skill_arg: str) -> Any:
        self._wait_time = int(skill_arg[0])
        self._internal_log(f"Requested wait time {self._wait_time}")

    def _is_skill_done(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> torch.Tensor:
        assert self._wait_time > 0
        return (self._cur_skill_step >= self._wait_time).float()

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        action = torch.zeros(prev_actions.shape, device=prev_actions.device)
        return action, rnn_hidden_states


class ResetArmSkill(SkillPolicy):
    def __init__(
        self,
        config,
        action_space: spaces.Space,
        batch_size,
    ):
        super().__init__(config, action_space, batch_size, True)
        self._target = np.array([float(x) for x in config.RESET_JOINT_STATE])

        self._ac_start = 0
        for k, space in action_space.items():
            if k != "ARM_ACTION":
                self._ac_start += get_num_actions(space)
            else:
                break

    def on_enter(
        self,
        skill_arg: List[str],
        batch_idx: int,
        observations,
        rnn_hidden_states,
        prev_actions,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ret = super().on_enter(
            skill_arg, batch_idx, observations, rnn_hidden_states, prev_actions
        )

        self._initial_delta = (
            self._target - observations["joint"].cpu().numpy()
        )

        return ret

    def _parse_skill_arg(self, skill_arg: str):
        return None

    def _is_skill_done(
        self, observations, rnn_hidden_states, prev_actions, masks
    ):
        current_joint_pos = observations["joint"].cpu().numpy()

        return (
            torch.as_tensor(
                np.abs(current_joint_pos - self._target).max(-1),
                device=rnn_hidden_states.device,
                dtype=torch.float32,
            )
            < 5e-2
        ).float()

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        current_joint_pos = observations["joint"].cpu().numpy()
        delta = self._target - current_joint_pos

        # Dividing by max initial delta means that the action will
        # always in [-1,1] and has the benefit of reducing the delta
        # amount was we converge to the target.
        delta = delta / np.maximum(
            self._initial_delta.max(-1, keepdims=True), 1e-5
        )

        action = torch.zeros_like(prev_actions)

        action[..., self._ac_start : self._ac_start + 7] = torch.from_numpy(
            delta
        ).to(device=action.device, dtype=action.dtype)

        return action, rnn_hidden_states


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
        # Mask out the release if the object is already held.
        is_holding = observations[IsHoldingSensor.cls_uuid].view(-1)
        for i in torch.nonzero(is_holding):
            # Do not release the object once it is held
            action[i, self._ac_start + self._ac_len - 1] = 1.0
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
        return action, hxs


class ArtObjSkillPolicy(NnSkillPolicy):
    def on_enter(
        self,
        skill_arg: List[str],
        batch_idx: int,
        observations,
        rnn_hidden_states,
        prev_actions,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        super().on_enter(
            skill_arg, batch_idx, observations, rnn_hidden_states, prev_actions
        )
        self._did_leave_start_zone = torch.zeros(
            self._batch_size, device=prev_actions.device
        )
        self._episode_start_resting_pos = observations[
            RelativeRestingPositionSensor.cls_uuid
        ]

    def _is_skill_done(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> torch.Tensor:

        cur_resting_pos = observations[RelativeRestingPositionSensor.cls_uuid]

        did_leave_start_zone = (
            torch.norm(
                cur_resting_pos - self._episode_start_resting_pos, dim=-1
            )
            > self._config.START_ZONE_RADIUS
        )
        self._did_leave_start_zone = torch.logical_or(
            self._did_leave_start_zone, did_leave_start_zone
        )

        cur_resting_dist = torch.norm(
            observations[RelativeRestingPositionSensor.cls_uuid], dim=-1
        )
        is_within_thresh = cur_resting_dist < self._config.AT_RESTING_THRESHOLD

        is_not_holding = 1 - observations[IsHoldingSensor.cls_uuid].view(-1)

        return (
            is_not_holding
            * is_within_thresh.float()
            * self._did_leave_start_zone.float()
        )

    def _parse_skill_arg(self, skill_arg):
        self._internal_log(f"Parsing skill argument {skill_arg}")
        return int(skill_arg[-1].split("|")[1])


class PlaceSkillPolicy(PickSkillPolicy):
    @dataclass(frozen=True)
    class PlaceSkillArgs:
        obj: int
        targ: int

    def _get_multi_sensor_index(self, batch_idx: int, sensor_name: str) -> int:
        return self._cur_skill_args[batch_idx].targ

    def _mask_pick(self, action, observations):
        # Mask out the grasp if the object is already released.
        is_not_holding = 1 - observations[IsHoldingSensor.cls_uuid].view(-1)
        for i in torch.nonzero(is_not_holding):
            # Do not regrasp the object once it is released.
            action[i, self._ac_start + self._ac_len - 1] = -1.0
        return action

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
                f"Terminating with {rel_resting_pos} and {is_not_holding}",
                observations,
            )
        return is_done

    def _parse_skill_arg(self, skill_arg):
        obj = int(skill_arg[0].split("|")[1])
        targ = int(skill_arg[1].split("|")[1])
        return PlaceSkillPolicy.PlaceSkillArgs(obj=obj, targ=targ)


class NavSkillPolicy(NnSkillPolicy):
    def __init__(
        self,
        wrap_policy,
        config,
        action_space: spaces.Space,
        filtered_obs_space: spaces.Space,
        filtered_action_space: spaces.Space,
        batch_size,
    ):
        super().__init__(
            wrap_policy,
            config,
            action_space,
            filtered_obs_space,
            filtered_action_space,
            batch_size,
            should_keep_hold_state=True,
        )

    def _is_skill_done(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> torch.Tensor:
        filtered_prev_actions = prev_actions[
            :, self._ac_start : self._ac_start + self._ac_len
        ]

        lin_vel, ang_vel = (
            filtered_prev_actions[:, 0],
            filtered_prev_actions[:, 1],
        )
        should_stop = (
            torch.abs(lin_vel) < self._config.LIN_SPEED_STOP
            and torch.abs(ang_vel) < self._config.ANG_SPEED_STOP
        )
        return should_stop.float()

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
        self._nav_targs = [None for _ in range(batch_size)]
        self._is_at_targ = torch.zeros(batch_size)

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
        self._is_at_targ[batch_idx] = 0.0
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
                self._is_at_targ[i] = 1.0

            full_action[
                i, self._ac_start : self._ac_start + self._ac_len
            ] = torch.tensor(vel).to(masks.device)

        return full_action, rnn_hidden_states
