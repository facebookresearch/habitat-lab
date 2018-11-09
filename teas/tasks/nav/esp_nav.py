import teas
from teas.wrappers import EnvTimeLimit


class EspNavToy(teas.EmbodiedTask):
    def __init__(self, config):
        # TODO(akadian): abstract out the environment creation method,
        # currently it is not clear whether having an abstraction for TeasEnv
        # is needed. Eventually we will have a make method similar to Task
        self._env = EnvTimeLimit(teas.TeasEnv(config),
                                 max_episode_steps=config.max_episode_steps)
        self.seed(config.seed)
    
    def episodes(self):
        target_object = None
        yield target_object, self._env
    
    def seed(self, seed):
        # TODO(akadian): call env seed, currently not implemented for ESP
        pass
    
    def close(self):
        self._env.close()
