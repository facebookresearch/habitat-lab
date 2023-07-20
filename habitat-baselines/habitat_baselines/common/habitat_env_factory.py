# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
from typing import TYPE_CHECKING, Any, List, Type

from habitat import ThreadedVectorEnv, VectorEnv, logger, make_dataset
from habitat.config import read_write
from habitat.gym import make_gym_from_config
from habitat_baselines.common.env_factory import VectorEnvFactory

if TYPE_CHECKING:
    from omegaconf import DictConfig


class HabitatVectorEnvFactory(VectorEnvFactory):
    def construct_envs(
        self,
        config: "DictConfig",
        workers_ignore_signals: bool = False,
        enforce_scenes_greater_eq_environments: bool = False,
        is_first_rank: bool = True,
    ) -> VectorEnv:
        r"""Create VectorEnv object with specified config and env class type.
        To allow better performance, dataset are split into small ones for
        each individual env, grouped by scenes.
        """

        num_environments = config.habitat_baselines.num_environments
        configs = []
        dataset = make_dataset(config.habitat.dataset.type)
        scenes = config.habitat.dataset.content_scenes
        if "*" in config.habitat.dataset.content_scenes:
            scenes = dataset.get_scenes_to_load(config.habitat.dataset)

        if num_environments < 1:
            raise RuntimeError("num_environments must be strictly positive")

        if len(scenes) == 0:
            raise RuntimeError(
                "No scenes to load, multiple process logic relies on being able to split scenes uniquely between processes"
            )

        random.shuffle(scenes)

        scene_splits: List[List[str]] = [[] for _ in range(num_environments)]
        if len(scenes) < num_environments:
            msg = f"There are less scenes ({len(scenes)}) than environments ({num_environments}). "
            if enforce_scenes_greater_eq_environments:
                logger.warn(
                    msg
                    + "Reducing the number of environments to be the number of scenes."
                )
                num_environments = len(scenes)
                scene_splits = [[s] for s in scenes]
            else:
                logger.warn(
                    msg
                    + "Each environment will use all the scenes instead of using a subset."
                )
            for scene in scenes:
                for split in scene_splits:
                    split.append(scene)
        else:
            for idx, scene in enumerate(scenes):
                scene_splits[idx % len(scene_splits)].append(scene)
            assert sum(map(len, scene_splits)) == len(scenes)

        for env_index in range(num_environments):
            proc_config = config.copy()
            with read_write(proc_config):
                task_config = proc_config.habitat
                task_config.seed = task_config.seed + env_index
                remove_measure_names = []
                if not is_first_rank:
                    # Filter out non rank0_measure from the task config if we are not on rank0.
                    remove_measure_names.extend(
                        task_config.task.rank0_measure_names
                    )
                if (env_index != 0) or not is_first_rank:
                    # Filter out non-rank0_env0 measures from the task config if we
                    # are not on rank0 env0.
                    remove_measure_names.extend(
                        task_config.task.rank0_env0_measure_names
                    )

                task_config.task.measurements = {
                    k: v
                    for k, v in task_config.task.measurements.items()
                    if k not in remove_measure_names
                }

                if len(scenes) > 0:
                    task_config.dataset.content_scenes = scene_splits[
                        env_index
                    ]

            configs.append(proc_config)

        vector_env_cls: Type[Any]
        if int(os.environ.get("HABITAT_ENV_DEBUG", 0)):
            logger.warn(
                "Using the debug Vector environment interface. Expect slower performance."
            )
            vector_env_cls = ThreadedVectorEnv
        else:
            vector_env_cls = VectorEnv

        envs = vector_env_cls(
            make_env_fn=make_gym_from_config,
            env_fn_args=tuple((c,) for c in configs),
            workers_ignore_signals=workers_ignore_signals,
        )

        if config.habitat.simulator.renderer.enable_batch_renderer:
            envs.initialize_batch_renderer(config)

        return envs
