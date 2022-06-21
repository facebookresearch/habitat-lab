from gym import spaces
from gym.vector import VectorEnv, VectorEnvWrapper


class VectorEnvActionDictWrapper(VectorEnvWrapper):
    ACTION_KEY = "action"

    def __init__(self, env: VectorEnv):
        super().__init__(env)
        self._requires_dict = False
        if isinstance(self.action_space, spaces.Box):
            self._requires_dict = True
            self.action_space = spaces.Dict(
                {
                    self.ACTION_KEY: spaces.Dict(
                        {self.ACTION_KEY: self.action_space}
                    )
                }
            )

    def step_async(self, actions):
        if self._requires_dict:
            actions = actions[self.ACTION_KEY][self.ACTION_KEY]
        return self.env.step_async(actions)
