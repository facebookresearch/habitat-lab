from typing import List, Tuple, Union

import numpy as np
from gym import spaces
from gym.core import ObsType
from gym.vector import AsyncVectorEnv, VectorEnv, VectorEnvWrapper

from habitat.utils.gym_adapter import HabGymWrapper


class VectorEnvCloseAtWrapper(VectorEnvWrapper):
    # TODO : make this AsyncVectorEnv
    def __init__(self, env: VectorEnv):
        super().__init__(env)
        # _has_closed_envs is true if some environments closed. If it is false,
        # we can skip some of the computation when resetting or stepping.
        self._has_closed_envs = False

        self.open_envs = list(range(self.num_envs))

    @staticmethod
    def _remove_closed_obs(obs, open_envs):
        """
        We are forced to remove the observations of closed environments since AsyncVectorEnv
        does not do it automatically.
        """
        if isinstance(obs, np.ndarray):
            return np.take(obs, open_envs, axis=0)
        elif isinstance(obs, dict):
            return {
                k: VectorEnvCloseAtWrapper._remove_closed_obs(v, open_envs)
                for k, v in obs.items()
            }

    def reset_wait(
        self,
        return_info: bool = False,
        *args,
        **kwargs,
    ) -> Union[ObsType, Tuple[ObsType, List[dict]]]:
        if not self._has_closed_envs:
            return self.env.reset_wait(
                return_info=return_info, *args, **kwargs
            )
        if return_info:
            obs, info = self.env.reset_wait(
                return_info=return_info, *args, **kwargs
            )
            obs = VectorEnvCloseAtWrapper._remove_closed_obs(
                obs, self.open_envs
            )
            return obs, info
        else:
            obs = self.env.reset_wait(return_info=return_info, *args, **kwargs)
            obs = VectorEnvCloseAtWrapper._remove_closed_obs(
                obs, self.open_envs
            )
            return obs

    def step_wait(
        self, *args, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        if not self._has_closed_envs:
            return self.env.step_wait(*args, **kwargs)
        obs, rew, done, info = self.env.step_wait(*args, **kwargs)
        obs = VectorEnvCloseAtWrapper._remove_closed_obs(obs, self.open_envs)
        return obs, rew, done, info

    def close_at(self, index: int) -> None:
        r"""Closes this env and destroys it.

        :param index: which env to close. All indexes after this one will be
            shifted down by one.

        This is useful for not needing to call close on all environments when
        only some are active.
        """
        pipe = self.env.parent_pipes.pop(index)
        pipe.send(("close", None))
        pipe.recv()
        self.num_envs -= 1
        self.open_envs.pop(index)
        self._has_closed_envs = True
