#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Union

import gym
import numpy as np
from gym import spaces

from habitat.core.simulator import Observations
from habitat.core.spaces import EmptySpace
from habitat.utils.visualizations.utils import observations_to_image

try:
    import pygame

except ImportError:
    pygame = None


def flatten_dict(d, parent_key=""):
    # From https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    items = []
    for k, v in d.items():
        new_key = parent_key + "." + str(k) if parent_key else str(k)
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
        total_dim = sum(shape[0] for shape in obs_shapes)

        return spaces.Box(
            shape=(total_dim,), low=-1.0, high=1.0, dtype=np.float32
        )
    return spaces.Dict(
        {k: v for k, v in obs_space.spaces.items() if k in limit_keys}
    )


def _is_continuous(original_space: gym.Space) -> bool:
    """
    returns true if the original space is only suitable for continuous control
    """
    if isinstance(original_space, spaces.Box):
        # Any Box means it is continuous
        return True
    if isinstance(original_space, EmptySpace):
        return False
    if isinstance(original_space, Mapping):
        return any((_is_continuous(v) for v in original_space.values()))
    raise NotImplementedError(
        f"Unknow action space found : {original_space}. Can only be Box or Empty"
    )


def _recursive_continuous_size_getter(
    original_space: gym.Space, low: List[float], high: List[float]
):
    """
    Returns the size of a continuous action vector from a habitat environment action space
    """
    if isinstance(original_space, spaces.Box):
        assert len(original_space.shape) == 1
        low.extend(original_space.low.tolist())
        high.extend(original_space.high.tolist())
    elif isinstance(original_space, EmptySpace):
        low.append(-1.0)
        high.append(1.0)
    elif isinstance(original_space, Mapping):
        for v in original_space.values():
            _recursive_continuous_size_getter(v, low, high)
    else:
        raise NotImplementedError(
            f"Unknow continuous action space found : {original_space}. Can only be Box, Empty or Dict."
        )


def create_action_space(original_space: gym.Space) -> gym.Space:
    """
    Converts a habitat task action space into a either continuous (Box) or discrete (Discrete) gym.space.
    """
    assert isinstance(
        original_space, Mapping
    ), f"The action space of the environment needs to be a Mapping, but was {original_space}"
    if _is_continuous(original_space):
        low: List[float] = []
        high: List[float] = []
        _recursive_continuous_size_getter(original_space, low, high)
        return spaces.Box(
            low=np.array(low), high=np.array(high), dtype=np.float32
        )
    else:
        # discrete case. The ActionSpace class gives us the correct action size
        return spaces.Discrete(len(original_space))


def continuous_vector_action_to_hab_dict(
    original_action_space: spaces.Space,
    vector_action_space: spaces.Box,
    action: np.ndarray,
) -> Dict[str, Any]:
    """
    Converts a np.ndarray vector action into a habitat-lab compatible action dictionary.
    """
    # Assume that the action space only has one root SimulatorTaskAction
    root_action_names = tuple(original_action_space.spaces.keys())
    if len(root_action_names) == 1:
        # No need for a tuple if there is only one action
        root_action_names = root_action_names[0]
    action_name_to_lengths = {}
    for outer_k, act_dict in original_action_space.spaces.items():
        if isinstance(act_dict, EmptySpace):
            action_name_to_lengths[outer_k] = 1
        else:
            for k, v in act_dict.items():
                # The only element in the action
                action_name_to_lengths[k] = v.shape[0]

    # Determine action arguments for root_action_name
    action_args = {}
    action_offset = 0
    for action_name, action_length in action_name_to_lengths.items():
        action_values = action[action_offset : action_offset + action_length]
        action_args[action_name] = action_values
        action_offset += action_length

    action_dict = {
        "action": root_action_names,
        "action_args": action_args,
    }

    return action_dict


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
    - `OBS_KEYS`: Which observation names from the wrapped environment
      to include. The order of the key names is kept in the output observation
      array. If not specified, all observations are included.
    - `DESIRED_GOAL_KEYS`: By default is an empty list. If not empty,
      any observations are returned in the `desired_goal` returned key of the
      observation.
    - `ACTION_KEYS`: Include a subset of the allowed actions in the
      wrapped environment. If not specified, all actions are included.
    Example usage:
    """

    def __init__(self, env, save_orig_obs: bool = False):
        gym_config = env.config.GYM
        self._gym_goal_keys = gym_config.DESIRED_GOAL_KEYS
        self._gym_achieved_goal_keys = gym_config.ACHIEVED_GOAL_KEYS
        self._gym_action_keys = gym_config.ACTION_KEYS
        self._gym_obs_keys = gym_config.OBS_KEYS

        if self._gym_obs_keys is None:
            self._gym_obs_keys = list(env.observation_space.spaces.keys())
        if self._gym_action_keys is None:
            self._gym_action_keys = list(env.action_space.spaces.keys())

        self._last_obs: Optional[Observations] = None
        self._save_orig_obs = save_orig_obs
        self.orig_obs = None

        # Filtering the action spaces keys
        action_space = spaces.Dict(
            {
                k: v
                for k, v in env.action_space.spaces.items()
                if (
                    (self._gym_action_keys is None)
                    or (k in self._gym_action_keys)
                )
            }
        )

        self.original_action_space = action_space

        self.action_space = create_action_space(action_space)

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

        self._screen = None
        self._env = env

    def step(self, action: Union[np.ndarray, int]):
        assert self.action_space.contains(
            action
        ), f"Unvalid action {action} for action space {self.action_space}"
        if isinstance(self.action_space, spaces.Box):
            assert isinstance(action, np.ndarray)
            hab_action = continuous_vector_action_to_hab_dict(
                self.original_action_space, self.action_space, action
            )
        else:
            hab_action = {"action": action}
        return self._direct_hab_step(hab_action)

    @property
    def number_of_episodes(self) -> int:
        return self._env.number_of_episodes

    def current_episode(self, all_info: bool = False) -> int:
        """
        Returns the current episode of the environment.
        :param all_info: If true, all of the information in the episode
        will be provided. Otherwise, only episode_id and scene_id will
        be included
        :return: The BaseEpisode object for the current episode
        """
        return self._env.current_episode(all_info)

    def _direct_hab_step(self, action: Union[int, str, Dict[str, Any]]):
        obs, reward, done, info = self._env.step(action=action)
        self._last_obs = obs
        obs = self._transform_obs(obs)
        info = flatten_dict(info)
        return obs, reward, done, info

    def _transform_obs(self, obs):
        if self._save_orig_obs:
            self.orig_obs = obs

        observation = {
            "observation": OrderedDict(
                [(k, obs[k]) for k in self._gym_obs_keys]
            )
        }

        if len(self._gym_goal_keys) > 0:
            observation["desired_goal"] = OrderedDict(
                [(k, obs[k]) for k in self._gym_goal_keys]
            )

        if len(self._gym_achieved_goal_keys) > 0:
            observation["achieved_goal"] = OrderedDict(
                [(k, obs[k]) for k in self._gym_achieved_goal_keys]
            )

        for k, v in observation.items():
            if isinstance(self.observation_space, spaces.Box):
                observation[k] = np.concatenate(list(v.values()))
        if len(observation) == 1:
            observation = observation["observation"]

        return observation

    def reset(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        obs = self._env.reset()
        self._last_obs = obs
        return self._transform_obs(obs)

    def render(self, mode: str = "human") -> np.ndarray:
        frame = None
        if mode == "rgb_array":
            frame = observations_to_image(
                self._last_obs, self._env._env.get_metrics()
            )
        elif mode == "human":
            if pygame is None:
                raise ValueError(
                    "Render mode human not supported without pygame."
                )
            frame = observations_to_image(
                self._last_obs, self._env._env.get_metrics()
            )
            if self._screen is None:
                pygame.init()
                self._screen = pygame.display.set_mode(
                    [frame.shape[1], frame.shape[0]]
                )
            draw_frame = np.transpose(frame, (1, 0, 2))
            draw_frame = pygame.surfarray.make_surface(draw_frame)
            self._screen.fill((0, 0, 0))  # type: ignore[attr-defined]
            self._screen.blit(draw_frame, (0, 0))  # type: ignore[attr-defined]
            pygame.display.update()
        else:
            raise ValueError(f"Render mode {mode} not currently supported.")

        return frame

    def close(self):
        del self._last_obs
        if self._screen is not None:
            pygame.quit()  # type: ignore[unreachable]
        self._env.close()
