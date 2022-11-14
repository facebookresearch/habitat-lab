from typing import List, Dict
import random

from habitat import Config
from habitat import make_dataset
from habitat.core.dataset import ALL_SCENES_MASK
from habitat.core.vector_env import VectorEnv

from .eval_env_wrapper import EvalEnvWrapper


def _get_env_gpus(config: Config, rank: int) -> List[int]:
    """Get GPUs assigned to environments of a particular agent process."""
    num_agent_processes = 1
    num_env_gpus = len(config.EVAL_VECTORIZED.simulator_gpu_ids)
    num_env_gpus_per_agent_process = num_env_gpus // num_agent_processes
    assert num_agent_processes > 0
    assert (
        num_env_gpus >= num_agent_processes and num_env_gpus % num_agent_processes == 0
    )

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        return [lst[i : i + n] for i in range(0, len(lst), n)]

    gpus = chunks(
        config.EVAL_VECTORIZED.simulator_gpu_ids, num_env_gpus_per_agent_process
    )[rank]
    return gpus


def make_vector_envs(config: Config, max_scene_repeat_episodes: int = -1) -> VectorEnv:
    """Create vectorized environments and split scenes across environments.
    Arguments:
        max_scene_repeat_episodes: if > 0, this is the maximum number of
         consecutive episodes in the same scene — set to 1 to get some
         scene diversity in visualization but keep to -1 default for
         training as switching scenes adds overhead to the simulator
    """
    gpus = _get_env_gpus(config, rank=0)
    num_gpus = len(gpus)
    num_envs = config.NUM_ENVIRONMENTS
    assert num_envs >= num_gpus and num_envs % num_gpus == 0
    num_envs_per_gpu = num_envs // num_gpus

    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    if ALL_SCENES_MASK in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)

    if num_envs > 1:
        if len(scenes) == 0:
            raise RuntimeError("No scenes to load")
        elif len(scenes) < num_envs and len(scenes) != 1:
            raise RuntimeError("Not enough scenes for envs")
        random.shuffle(scenes)

    if len(scenes) == 1:
        scene_splits = [[scenes[0]] for _ in range(num_envs)]
    else:
        scene_splits = [[] for _ in range(num_envs)]
        for idx, scene in enumerate(scenes):
            scene_splits[idx % len(scene_splits)].append(scene)
        assert sum(map(len, scene_splits)) == len(scenes)

    configs = []
    for i in range(num_gpus):
        for j in range(num_envs_per_gpu):
            proc_config = config.clone()
            proc_config.defrost()
            proc_id = (i * num_envs_per_gpu) + j
            task_config = proc_config.TASK_CONFIG
            task_config.SEED += proc_id
            task_config.DATASET.CONTENT_SCENES = scene_splits[proc_id]
            if not proc_config.NO_GPU:
                task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpus[i]
            task_config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
            task_config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = (
                max_scene_repeat_episodes
            )
            proc_config.freeze()
            configs.append(proc_config)

    envs = VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple([(configs[rank],) for rank in range(len(configs))]),
    )
    return envs


def make_vector_envs_on_specific_episodes(
    config: Config, scene2episodes: Dict[str, List[str]]
) -> VectorEnv:
    """Create vectorized environments to evaluate specific episodes.
    Arguments:
        scene2episodes: mapping from scene ID to episode IDs
    """
    scenes = list(scene2episodes.keys())

    gpus = _get_env_gpus(config, rank=0)
    num_gpus = len(gpus)
    num_envs = len(scenes)  # One environment per scene

    configs = []
    episode_ids = []
    for proc_id in range(num_envs):
        proc_config = config.clone()
        proc_config.defrost()
        task_config = proc_config.TASK_CONFIG
        task_config.SEED += proc_id
        task_config.DATASET.CONTENT_SCENES = [scenes[proc_id]]
        if not proc_config.NO_GPU:
            task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpus[
                proc_id % num_gpus
            ]
        task_config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        proc_config.freeze()
        configs.append(proc_config)
        episode_ids.append(scene2episodes[scenes[proc_id]])

    envs = VectorEnv(
        make_env_fn=make_env_on_specific_episodes_fn,
        env_fn_args=tuple(
            [(configs[rank], episode_ids[rank]) for rank in range(len(configs))]
        ),
    )
    return envs


def make_env_fn(config):
    return EvalEnvWrapper(config)


def make_env_on_specific_episodes_fn(config, episode_ids):
    return EvalEnvWrapper(config, episode_ids)
