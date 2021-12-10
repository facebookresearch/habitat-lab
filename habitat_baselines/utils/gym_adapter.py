#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, Union

import gym
import numpy as np
from gym import spaces

from habitat.core.simulator import Observations
from habitat.utils.visualizations.utils import observations_to_image


def flatten_dict(d, parent_key=""):
    # From https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    items = []
    for k, v in d.items():
        new_key = parent_key + str(k) if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def smash_observation_space(obs_space, limit_keys):
    obs_shapes = [obs_space.spaces[k].shape for k in limit_keys]

    def transform_shape(shape):
        if len(shape) == 2:
            return (np.prod(shape),)
        return shape

    obs_shapes = [transform_shape(shape) for shape in obs_shapes]
    obs_dims = [len(shape) for shape in obs_shapes]
    if len(set(obs_dims)) == 1 and obs_dims[0] == 1:
        # Smash together
        total_dim = sum([shape[0] for shape in obs_shapes])

        return spaces.Box(
            shape=(total_dim,), low=-1.0, high=1.0, dtype=np.float32
        )
    return obs_space


class HabGymWrapper(gym.Env):
    """
    Wraps a Habitat RLEnv into a format compatible with the standard OpenAI Gym
    interface. Currently does not support discrete actions. This wrapper
    therefore changes the behavior so that:
    - The action input to `.step(...)` is always a numpy array
    - The returned value of `.step(...)` and `.reset()` is a either a numpy array or a
      dictionary consisting of string keys and numpy array values.
    - The action space is converted to a `gym.spaces.Box`, action spaces from the RLEnv are
      flattened into one Box space.
    - The observation space is either a `gym.spaces.Box` or a `gym.spaces.Dict`
      where the spaces of the Dict are `gym.spaces.Box`.
    Configuration allows filtering the included observations, specifying goals,
    or filtering actions. Listed below are the
    config keys:
    - `RL.GYM_OBS_KEYS`: Which observation names from the wrapped environment
      to include. The order of the key names is kept in the output observation
      array. If not specified, all observations are included.
    - `RL.GYM_DESIRED_GOAL_KEYS`: By default is an empty list. If not empty,
      any observations are returned in the `desired_goal` returned key of the
      observation.
    - `RL.GYM_FIX_INFO_DICT`: By default True, but if specified as true, this
      flattens the returned info dictionary to have depth 1 where sub-keys are
      concatenated to parent keys.
    - `RL.GYM_ACTION_KEYS`: Include a subset of the allowed actions in the
      wrapped environment. If not specified, all actions are included.
    Example usage:
    ```
    config = baselines_get_config(hab_cfg_path)
    env_class = get_env_class(config.ENV_NAME)

    env = habitat_baselines.utils.env_utils.make_env_fn(
        env_class=env_class, config=config
    )
    env = HabGymWrapper(env)
    env = HabRenderWrapper(env)
    ```
    """

    def __init__(self, env, save_orig_obs: bool = False):
        self._gym_goal_keys = env._rl_config.get("GYM_DESIRED_GOAL_KEYS", [])
        self._gym_achieved_goal_keys = env._rl_config.get(
            "GYM_ACHIEVED_GOAL_KEYS", []
        )
        self._fix_info_dict = env._rl_config.get("GYM_FIX_INFO_DICT", True)
        self._gym_action_keys = env._rl_config.get("GYM_ACTION_KEYS", None)
        self._gym_obs_keys = env._rl_config.get("GYM_OBS_KEYS", None)

        if self._gym_obs_keys is None:
            self._gym_obs_keys = list(env.observation_space.spaces.keys())
        if self._gym_action_keys is None:
            self._gym_action_keys = list(env.action_space.spaces.keys())

        action_space = env.action_space
        action_space = spaces.Dict(
            {
                k: v
                for k, v in action_space.spaces.items()
                if (
                    (self._gym_action_keys is None)
                    or (k in self._gym_action_keys)
                )
            }
        )
        self._last_obs: Optional[Observations] = None
        self.action_mapping = {}
        self._save_orig_obs = save_orig_obs
        self.orig_obs = None
        if len(action_space.spaces) != 1:
            raise ValueError(
                "Cannot convert this action space, more than one action"
            )

        self.orig_action_name = list(action_space.spaces.keys())[0]
        action_space = action_space.spaces[self.orig_action_name]
        if not isinstance(action_space, spaces.Dict):
            raise ValueError("Cannot convert this action space")

        all_box = True
        for sub_space in action_space.spaces.values():
            if not isinstance(sub_space, spaces.Box):
                all_box = False
                break
        if not all_box:
            raise ValueError("Cannot convert this action space")
        start_i = 0
        for name, sub_space in action_space.spaces.items():
            end_i = start_i + sub_space.shape[0]
            self.action_mapping[name] = (start_i, end_i)
            start_i = end_i

        self.action_space = spaces.Box(
            shape=(end_i,), low=-1.0, high=1.0, dtype=np.float32
        )

        self.observation_space = smash_observation_space(
            env.observation_space, self._gym_obs_keys
        )

        dict_space = {
            "observation": self.observation_space,
        }

        if len(self._gym_goal_keys) > 0:
            dict_space["desired_goal"] = smash_observation_space(
                env.observation_space, self._gym_goal_keys
            )

        if len(self._gym_achieved_goal_keys) > 0:
            dict_space["achieved_goal"] = smash_observation_space(
                env.observation_space, self._gym_achieved_goal_keys
            )

        if len(dict_space) > 1:
            self.observation_space = spaces.Dict(dict_space)

        self._env = env

    def step(self, action: np.ndarray):
        action_args = {}
        for k, (start_i, end_i) in self.action_mapping.items():
            action_args[k] = action[start_i:end_i]
        action = {
            "action": self.orig_action_name,
            "action_args": action_args,
        }
        return self.direct_hab_step(action)

    def direct_hab_step(self, action: Union[int, str, Dict[str, Any]]):
        obs, reward, done, info = self._env.step(action=action)
        self._last_obs = obs
        obs = self._transform_obs(obs)
        if self._fix_info_dict:
            info = flatten_dict(info)
            info = {k: float(v) for k, v in info.items()}

        return obs, reward, done, info

    def _is_space_flat(self, space_name):
        if isinstance(self.observation_space, spaces.Box):
            return True
        return isinstance(
            self.observation_space.spaces[space_name], spaces.Box
        )

    def _transform_obs(self, obs):
        if self._save_orig_obs:
            self.orig_obs = obs
        observation = {"observation": [obs[k] for k in self._gym_obs_keys]}

        if len(self._gym_goal_keys) > 0:
            observation["desired_goal"] = [obs[k] for k in self._gym_goal_keys]

        if len(self._gym_achieved_goal_keys) > 0:
            observation["achieved_goal"] = [
                obs[k] for k in self._gym_achieved_goal_keys
            ]

        for k, v in observation.items():
            if self._is_space_flat(k):
                observation[k] = np.concatenate(v)
        if len(observation) == 1:
            return observation["observation"]

        return observation

    def reset(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        obs = self._env.reset()
        self._last_obs = obs
        return self._transform_obs(obs)

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        frame = None
        if mode == "rgb_array":
            frame = observations_to_image(
                self._last_obs, self._env._env.get_metrics()
            )
        else:
            raise ValueError(f"Render mode {mode} not currently supported.")

        return frame

    def close(self):
        self._env.close()
