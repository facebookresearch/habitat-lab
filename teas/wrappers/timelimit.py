import time

import teas


class EnvTimeLimit(teas.EnvWrapper):
    def __init__(self, env, max_episode_seconds=None, max_episode_steps=None):
        super().__init__(env)
        self._max_episode_seconds = max_episode_seconds
        self._max_episode_steps = max_episode_steps
        
        self._elapsed_steps = 0
        self._episode_started_at = None
    
    @property
    def _elapsed_seconds(self):
        return time.time() - self._episode_started_at
    
    def _past_limit(self):
        if self._max_episode_steps is not None and self._max_episode_steps <= \
                self._elapsed_steps:
            return True
        elif self._max_episode_seconds is not None and \
                self._max_episode_seconds <= self._elapsed_seconds:
            return True
        return False
    
    def step(self, action):
        assert self._episode_started_at is not None, "Cannot call step() " \
                                                     "before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        
        if self._past_limit():
            done = True
        
        return observation, reward, done, info
    
    def reset(self):
        self._episode_started_at = time.time()
        self._elapsed_steps = 0
        return self.env.reset()
