from gym import spaces
from gym.vector import VectorEnv, VectorEnvWrapper
from typing import List, Optional, Sequence, Tuple, Union
from gym.core import ObsType

import numpy as np

#from https://github.com/openai/gym/blob/4d57b864f866b5f38ce8a05024184d675c897727/gym/vector/vector_env.py#L295

class VectorEnvObsDictWrapper(VectorEnvWrapper):
    OBSERVATION_KEY = "obs"

    def __init__(self, env: VectorEnv):
        super().__init__(env)
        self._requires_dict = False
        if isinstance(self.observation_space, spaces.Box):
            self._requires_dict = True
            self.observation_space = spaces.Dict(
                {
                        self.OBSERVATION_KEY: self.observation_space
                }
            )

    def step_wait(
            self, timeout: Optional[Union[int, float]] = None
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        obs, reward, done, info = self.env.step_wait(timeout)
        if self._requires_dict:
            obs ={self.OBSERVATION_KEY: obs}
        return  obs, reward, done, info
    
    def reset_wait(
        self,
        timeout: Optional[Union[int, float]] = None,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[ObsType, Tuple[ObsType, List[dict]]]:
        if return_info and self._requires_dict:
            obs, info = self.env.reset_wait(timeout, seed, return_info,options )
            return {self.OBSERVATION_KEY: obs}, info
        if not return_info and self._requires_dict:
            obs = self.env.reset_wait(timeout, seed, return_info,options )
            return {self.OBSERVATION_KEY: obs}
        return self.env.reset_wait(timeout, seed, return_info,options )

