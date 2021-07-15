import gym
import numpy as np
from gym import spaces

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


class HabGymWrapper(gym.Env):
    def __init__(self, env):
        action_space = env.action_space
        self._gym_obs_keys = env._rl_config.GYM_OBS_KEYS
        self._fix_info_dict = env._rl_config.GYM_FIX_INFO_DICT
        self._last_obs = None
        self.action_mapping = {}
        if len(action_space.spaces) != 1:
            raise ValueError("Cannot convert this action space")

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
        self.action_space = spaces.Box(
            shape=(end_i,), low=-1.0, high=1.0, dtype=np.float32
        )

        obs_shapes = [
            env.observation_space.spaces[k].shape for k in self._gym_obs_keys
        ]

        def transform_shape(shape):
            if len(shape) == 2:
                return (np.prod(shape),)
            return shape

        obs_shapes = [transform_shape(shape) for shape in obs_shapes]
        obs_dims = [len(shape) for shape in obs_shapes]
        self.combine_obs = False
        if len(set(obs_dims)) == 1 and obs_dims[0] == 1:
            self.combine_obs = True
            # Smash together
            total_dim = sum([shape[0] for shape in obs_shapes])
            self.observation_space = spaces.Box(
                shape=(total_dim,), low=-1.0, high=1.0, dtype=np.float32
            )

        self._env = env

    def step(self, action):
        action_args = {}
        for k, (start_i, end_i) in self.action_mapping.items():
            action_args[k] = action[start_i:end_i]
        action = {
            "action": self.orig_action_name,
            "action_args": action_args,
        }
        obs, reward, done, info = self._env.step(action=action)
        self._last_obs = obs
        obs = self._transform_obs(obs)
        if self._fix_info_dict:
            info = flatten_dict(info)
            info = {k: float(v) for k, v in info.items()}

        return obs, reward, done, info

    def _transform_obs(self, obs):
        obs = {k: v for k, v in obs.items() if k in self._gym_obs_keys}
        if self.combine_obs:
            obs = np.concatenate([obs[k] for k in self._gym_obs_keys])
        return obs

    def reset(self):
        obs = self._env.reset()
        self._last_obs = obs
        return self._transform_obs(obs)

    def render(self, mode="rgb_array"):
        frame = observations_to_image(
            self._last_obs, self._env._env.get_metrics()
        )

        return frame
