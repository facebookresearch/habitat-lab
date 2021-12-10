import copy
import os.path as osp

import habitat
from habitat.core.registry import registry

TASK_CONFIGS_DIR = "configs/tasks/rearrange/"


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
    task_cls = registry.get_task(task)

    config = copy.copy(cur_config)
    config.defrost()
    if task_def is not None:
        task_config = habitat.get_config(
            osp.join(TASK_CONFIGS_DIR, task_def + ".yaml")
        )
        config.merge_from_other_cfg(task_config.TASK)
    config.freeze()
    task = task_cls(config=config, dataset=cur_dataset, sim=cur_env._sim)

    task.set_args(**task_kwargs)
    task.set_sim_reset(should_super_reset)
    task.reset(episode)
    return task
