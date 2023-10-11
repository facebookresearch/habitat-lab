#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

import gym
import numpy as np
from gym import spaces

from habitat.core.batch_rendering.env_batch_renderer_constants import (
    KEYFRAME_OBSERVATION_KEY,
)
from habitat.core.simulator import Observations
from habitat.core.spaces import EmptySpace
from habitat.tasks.rearrange.rearrange_sim import add_perf_timing_func
from habitat.utils.visualizations.utils import observations_to_image

if TYPE_CHECKING:
    from habitat.core.dataset import BaseEpisode
    from habitat.core.env import RLEnv

try:
    import pygame

except ImportError:
    pygame = None

HabGymWrapperObsType = Union[np.ndarray, Dict[str, np.ndarray]]


def filter_observation_space(obs_space, limit_keys):
    return spaces.Dict(
        {k: v for k, v in obs_space.spaces.items() if k in limit_keys}
    )


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
    return filter_observation_space(obs_space, limit_keys)


def _is_continuous(original_space: gym.Space) -> bool:
    """
    returns true if the original space is only suitable for continuous control
    """
    if isinstance(original_space, spaces.Box):
        # Any Box means it is continuous
        return True
    if isinstance(original_space, EmptySpace):
        return False
    if isinstance(original_space, spaces.Discrete):
        return False
    if isinstance(original_space, Mapping):
        return any((_is_continuous(v) for v in original_space.values()))
    raise NotImplementedError(
        f"Unknown action space found : {original_space}. Can only be Box or Empty"
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
            f"Unknown continuous action space found : {original_space}. Can only be Box, Empty or Dict."
        )


def create_action_space(original_space: gym.Space) -> gym.Space:
    """
    Converts a habitat task action space into a either continuous (Box) or discrete (Discrete) gym.space.
    """
    assert isinstance(
        original_space, Mapping
    ), f"The action space of the environment needs to be a Mapping, but was {original_space}"
    first_k = list(original_space.keys())[0]
    if _is_continuous(original_space):
        low: List[float] = []
        high: List[float] = []
        _recursive_continuous_size_getter(original_space, low, high)
        return spaces.Box(
            low=np.array(low), high=np.array(high), dtype=np.float32
        )
    elif len(original_space) == 1 and isinstance(
        original_space[first_k], spaces.Discrete
    ):
        return spaces.Discrete(original_space[first_k].n)
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


class HabGymWrapper(gym.Wrapper):
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
    - `obs_keys`: Which observation names from the wrapped environment
      to include. The order of the key names is kept in the output observation
      array. If not specified, all observations are included.
    - `desired_goal_keys`: By default is an empty list. If not empty,
      any observations are returned in the `desired_goal` returned key of the
      observation.
    - `action_keys`: Include a subset of the allowed actions in the
      wrapped environment. If not specified, all actions are included.
    Example usage:
    """

    def __init__(
        self,
        env: "RLEnv",
        save_orig_obs: bool = False,
    ):
        super().__init__(env)

        habitat_gym_config = env.config.gym
        self._gym_goal_keys = habitat_gym_config.desired_goal_keys
        self._gym_achieved_goal_keys = habitat_gym_config.achieved_goal_keys
        self._gym_action_keys = habitat_gym_config.action_keys
        self._gym_obs_keys = habitat_gym_config.obs_keys

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

        self.observation_space = filter_observation_space(
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

        self._screen: Optional[pygame.surface.Surface] = None
        # Store so we can profile functions on this class.
        self._sim = self.env._env._sim

    @add_perf_timing_func()
    def step(
        self, action: Union[np.ndarray, int]
    ) -> Tuple[HabGymWrapperObsType, float, bool, dict]:
        assert self.action_space.contains(
            action
        ), f"Invalid action {action} for action space {self.action_space}"

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
        return self.env.number_of_episodes

    def current_episode(self, all_info: bool = False) -> "BaseEpisode":
        return self.env.current_episode(all_info)

    def _direct_hab_step(self, action: Union[int, str, Dict[str, Any]]):
        obs, reward, done, info = self.env.step(action=action)
        self._last_obs = obs
        obs = self._transform_obs(obs)
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

        if KEYFRAME_OBSERVATION_KEY in obs:
            observation[KEYFRAME_OBSERVATION_KEY] = obs[
                KEYFRAME_OBSERVATION_KEY
            ]

        for k, v in observation.items():
            if isinstance(self.observation_space, spaces.Box):
                observation[k] = np.concatenate(list(v.values()))
        if len(observation) == 1:
            observation = observation["observation"]

        return observation

    def reset(
        self, *args, return_info: bool = False, **kwargs
    ) -> Union[HabGymWrapperObsType, Tuple[HabGymWrapperObsType, dict]]:
        obs = self.env.reset(*args, return_info=return_info, **kwargs)
        if return_info:
            obs, info = obs
            self._last_obs = obs
            return self._transform_obs(obs), info
        else:
            self._last_obs = obs
            return self._transform_obs(obs)

    def render(self, mode: str = "human", **kwargs):
        last_infos = self.env.get_info(observations=None)
        if mode == "rgb_array":
            frame = observations_to_image(self._last_obs, last_infos)
        elif mode == "human":
            if pygame is None:
                raise ValueError(
                    "Render mode human not supported without pygame."
                )
            frame = observations_to_image(self._last_obs, last_infos)
            if self._screen is None:
                pygame.init()
                self._screen = pygame.display.set_mode(
                    [frame.shape[1], frame.shape[0]]
                )
            draw_frame = np.transpose(
                frame, (1, 0, 2)
            )  # (H, W, C) -> (W, H, C)
            draw_frame = pygame.surfarray.make_surface(draw_frame)
            black_color = (0, 0, 0)
            top_corner = (0, 0)
            self._screen.fill(color=black_color)  # type: ignore[attr-defined]
            self._screen.blit(draw_frame, dest=top_corner)  # type: ignore[attr-defined]
            pygame.display.update()
        else:
            raise ValueError(f"Render mode {mode} not currently supported.")

        return frame

    def close(self):
        del self._last_obs
        if self._screen is not None:
            pygame.quit()  # type: ignore[unreachable]
        self.env.close()

    @property
    def unwrapped(self) -> "RLEnv":
        return cast("RLEnv", self.env.unwrapped)
