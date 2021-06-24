import numpy as np

print("base hab env regs")
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import NavRLEnv


@baseline_registry.register_env(name="EmptyRLEnv")
class BaseHabEnv(NavRLEnv):
    """
    Defines the core structure allowing envs to easily define their own reward
    function, success condition and pre-step code.
    """

    def __init__(self, config, dataset=None):
        config.defrost()
        config.TASK_CONFIG.SIMULATOR.N_OBJS = len(dataset.episodes[0].targets)
        config.freeze()

        super().__init__(config, dataset)
        self.config = config
        self.prev_obs = None
        self.tcfg = config.TASK_CONFIG
        self.rlcfg = config.RL

    def _my_episode_success(self):
        """
        Override this
        """
        return False

    def get_task_obs(self):
        sim = self._env._sim
        sim._try_acquire_context()
        prev_sim_obs = sim.get_sensor_observations()
        obs = sim._sensor_suite.get_observations(prev_sim_obs)
        task_obs = self._env.task.sensor_suite.get_observations(obs, episode=0)
        obs.update(task_obs)
        self.prev_obs = obs
        return self._trans_obs(obs)

    def _trans_obs(self, obs):
        return obs

    def _episode_success(self):
        """
        Don't override this, there seems to be some side effect of this in the
        parent environment.
        """
        return False

    def get_reward(self, obs):
        """
        Override this
        """
        return 0

    def _my_get_reward(self, obs):
        """
        Don't override this, there seems to be some side effect of this in the
        parent environment.
        """
        return 0

    def _pre_step(self):
        pass

    def set_args(self):
        pass

    def step(self, action, action_args):
        obs, reward, done, info = super().step(action={'action': action,
            'action_args': action_args})
        self._pre_step()
        reward = self._my_get_reward(obs)
        add_info = {}
        for k, v in info.items():
            if isinstance(v, dict):
                add_info["ep_avg_" + k] = np.mean(list(v.values()))
            elif isinstance(v, list):
                add_info["ep_avg_" + k] = np.mean(v)
        info.update(add_info)
        info["ep_success"] = int(self._my_episode_success())

        if self._my_episode_success():
            done = True
        return obs, reward, done, info

    def reset(self):
        obs = super().reset()
        self.first_obs = obs
        return obs
