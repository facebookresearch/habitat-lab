# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Union

from gym import Env, Wrapper, spaces
from gym.core import ActType, ObsType


class EnvObsDictWrapper(Wrapper):
    OBSERVATION_KEY = "obs"
    observation_space: spaces.Space

    def __init__(self, env: Env):
        """
        Wraps a VectorEnv environment and makes sure its obervation space is a
        Dictionary (If it is a Box, it will be wrapped into a dictionary)
        """
        super().__init__(env)
        self._requires_dict = False
        if isinstance(self.observation_space, spaces.Box):
            self._requires_dict = True
            self.observation_space = spaces.Dict(
                {self.OBSERVATION_KEY: self.observation_space}
            )

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, reward, done, info = self.env.step(action)
        if self._requires_dict:
            obs = {self.OBSERVATION_KEY: obs}
        return obs, reward, done, info

    def reset(self, **kwargs) -> Union[ObsType, Tuple[ObsType, dict]]:
        if not self._requires_dict:
            return self.env.reset(**kwargs)
        reset_output = self.env.reset(**kwargs)
        if isinstance(reset_output, tuple):
            obs, info = self.env.reset(**kwargs)
            return {self.OBSERVATION_KEY: obs}, info
        else:
            obs = self.env.reset(**kwargs)
            return {self.OBSERVATION_KEY: obs}
