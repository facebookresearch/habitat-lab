import copy
import os.path as osp

import numpy as np
import yacs.config
import yaml

import habitat
from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.nav.nav import NavigationTask
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.tasks.rearrange.utils import CacheHelper, rearrange_collision

TASK_CONFIGS_DIR = "configs/tasks/rearrange/"


# class DummyTask(object):
#    def __init__(self, sim, config, dataset=None):
#        self.config = config
#        self._sim = sim
#        # self.observation_space = self._env.observation_space
#        # self.action_space = self._env.action_space
#        # self.number_of_episodes = self._env.number_of_episodes
#        # self.reward_range = self.get_reward_range()
#
#    def set_args(self, **kwargs):
#        pass
#
#    def reset(self):
#        # THIS IS THE KEY THING, DO NOTHING ON THE RESET
#        self._sim._try_acquire_context()
#        prev_sim_obs = self._sim.get_sensor_observations()
#        obs = self._sim._sensor_suite.get_observations(prev_sim_obs)
#        return obs


def load_task_object(
    task,
    task_def,
    cur_config,
    cur_env,
    cur_dataset,
    should_super_reset,
    task_kwargs,
    episode,
):
    # prev_base = NavigationTask.__bases__[0]
    # NavigationTask.__bases__ = (DummyTask,)
    task_cls = registry.get_task(task)

    # yacs.config._VALID_TYPES.add(type(cur_env))
    # task_config_name = task_def

    config = copy.copy(cur_config)
    config.defrost()
    if task_def is not None:
        task_config = habitat.get_config(
            osp.join(TASK_CONFIGS_DIR, task_def + ".yaml")
        )
        config.merge_from_other_cfg(task_config.TASK)
    # config.tmp_env = cur_env
    config.freeze()
    task = task_cls(config=config, dataset=cur_dataset, sim=cur_env._sim)
    # config.defrost()
    # del config["tmp_env"]
    # config.freeze()
    # yacs.config._VALID_TYPES.remove(type(cur_env._env))

    task.set_args(**task_kwargs)
    task.set_sim_reset(should_super_reset)
    task.reset(episode)
    # NavigationTask.__bases__ = (prev_base,)
    return task
