from typing import Tuple

import gym.spaces as spaces
import magnum as mn
import numpy as np
import torch
import torch.nn as nn

from habitat.core.spaces import ActionSpace
from habitat.tasks.rearrange.rearrange_sensors import (
    AbsTargetStartSensor,
    LocalizationSensor,
)
from habitat.tasks.rearrange.sub_tasks.nav_to_obj_sensors import (
    DistToNavGoalSensor,
    NavGoalSensor,
    OracleNavigationActionSensor,
)
from habitat.tasks.utils import get_angle
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.hrl.high_level_policy import (
    GtHighLevelPolicy,
    HighLevelPolicy,
)
from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.utils.common import get_num_actions


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

        self._cur_skill_args = torch.zeros(self._batch_size, dtype=torch.int32)

        for k in action_space:
            if k not in filtered_action_space.keys():
                self._ac_start += get_num_actions(action_space[k])
            else:
                break

    def _select_obs(self, obs, cur_batch_idx):
        for k in self._config.OBS_SKILL_INPUTS:
            entity_positions = obs[k].view(1, -1, 3)
            obs[k] = entity_positions[
                :, self._cur_skill_args[cur_batch_idx].item()
            ]
        return obs

    def should_terminate(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> torch.Tensor:
        return torch.zeros(observations.shape[0]).to(masks.device)

    def on_enter(
        self,
        skill_arg,
        batch_idx,
        observations,
        rnn_hidden_states,
        prev_actions,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :returns: The new hidden state and prev_actions ONLY at the batch_idx.
        """
        self._cur_skill_args[batch_idx] = skill_arg

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

    def act(
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
                k: v
                for k, v in observations.items()
                if k in self._filtered_obs_space.keys()
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
            batch_size,
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


class NavSkillPolicy(NnSkillPolicy):
    def should_terminate(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> torch.Tensor:
        if self._config.ORACLE_STOP:
            dist_to_nav_goal = observations[DistToNavGoalSensor.cls_uuid]
            should_stop = dist_to_nav_goal < self._config.ORACLE_STOP_DIST
            return should_stop.float()
        return torch.zeros(masks.shape[0]).to(masks.device)


class OracleNavPolicy(NnSkillPolicy):
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
        return self

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
        return cls(
            None,
            config,
            action_space,
            observation_space,
            action_space,
            batch_size,
        )

    def should_terminate(
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

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        batch_nav_targ = observations[OracleNavigationActionSensor.cls_uuid]
        batch_localization = observations[LocalizationSensor.cls_uuid]
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

            full_action[i, -2:] = torch.tensor(vel).to(masks.device)
        return full_action, rnn_hidden_states
