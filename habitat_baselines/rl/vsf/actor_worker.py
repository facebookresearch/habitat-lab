import queue
import random
from multiprocessing import Event
from multiprocessing.context import BaseContext
from typing import Any, List, Type, Union

import attr
import faster_fifo
import numpy as np
import torch

from habitat import Config, Env, RLEnv, VectorEnv, make_dataset
from habitat_baselines.rl.vsf.task_enums import (
    ActorWorkerTasks,
    ReportWorkerTasks,
)
from habitat_baselines.rl.vsf.timing import Timing
from habitat_baselines.rl.vsf.worker_common import ProcessBase, WorkerBase
from habitat_baselines.utils.env_utils import make_env_fn


def int_action_plugin(act):
    return {"action": {"action": int(act)}}


@attr.s(auto_attribs=True)
class ActorWorkerProcess(ProcessBase):
    actor_idx: int
    make_env_fn: Any
    env_fn_args: Any
    policy_worker_queue: faster_fifo.Queue
    report_queue: faster_fifo.Queue
    task_queue: faster_fifo.Queue
    action_plugin: Any
    done_event: Event
    env: RLEnv = attr.ib(default=None, init=False)
    total_reward: float = 0.0
    episode_length: float = 0.0
    timer: Timing = attr.Factory(Timing)

    def close(self):
        if self.env is None:
            self.env.close()

        self.env = None

    def start(self):
        assert self.env is None
        self.env: RLEnv = self.make_env_fn(*self.env_fn_args)

        self.report_queue.put(
            dict(
                actor_idx=self.actor_idx,
                obs_space=self.env.observation_space,
                act_space=self.env.action_space,
            )
        )

    def reset(self):
        obs = self.env.reset()
        self.transfer_buffers["observations"] = obs

        self.policy_worker_queue.put(self.actor_idx)

    def step(self, action):
        with self.timer.avg_time("Process-Actions"):
            action = self.action_plugin(action)

        with self.timer.avg_time("Env-Step"):
            obs, reward, done, info = self.env.step(**action)

        with self.timer.avg_time("Env-Reset"):
            if done:
                obs = self.env.reset()

        with self.timer.avg_time("Env-Enqueue"):
            self.transfer_buffers["observations"] = obs
            self.transfer_buffers["rewards"] = np.array(
                reward, dtype=np.float32
            )
            self.transfer_buffers["masks"] = np.array(not done, dtype=np.bool)

        self.policy_worker_queue.put(self.actor_idx)

        self.total_reward += reward
        self.episode_length += 1.0

        if done:
            self.report_queue.put_many(
                [
                    (
                        ReportWorkerTasks.episode_end,
                        dict(
                            actor_idx=self.actor_idx,
                            length=self.episode_length,
                            reward=self.total_reward,
                            info=info,
                        ),
                    ),
                    (ReportWorkerTasks.actor_timing, self.timer),
                ]
            )
            self.total_reward = 0.0
            self.episode_length = 0.0

    def run(self):
        while not self.done_event.is_set():
            try:
                task, data = self.task_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if task == ActorWorkerTasks.step:
                self.step(data)
            elif task == ActorWorkerTasks.start:
                self.start()
            elif task == ActorWorkerTasks.reset:
                self.reset()
            elif task == ActorWorkerTasks.set_transfer_buffers:
                self.transfer_buffers = data[self.actor_idx]
            else:
                raise RuntimeError(f"Unknown task {task}")

        self.close()


class ActorWorker(WorkerBase):
    def __init__(
        self,
        mp_ctx: BaseContext,
        actor_idx: int,
        make_env_fn,
        env_fn_args,
        policy_worker_queue: faster_fifo.Queue,
        report_queue: faster_fifo.Queue,
        actor_worker_queue: faster_fifo.Queue,
        action_plugin=int_action_plugin,
    ):
        super().__init__(
            mp_ctx,
            ActorWorkerProcess,
            actor_idx,
            make_env_fn,
            env_fn_args,
            policy_worker_queue,
            report_queue,
            actor_worker_queue,
            action_plugin,
        )
        self.actor_worker_queue = actor_worker_queue

    def start(self):
        self.actor_worker_queue.put((ActorWorkerTasks.start, None))

    def reset(self):
        self.actor_worker_queue.put((ActorWorkerTasks.reset, None))

    def set_transfer_buffers(self, buffers):
        self.actor_worker_queue.put(
            (ActorWorkerTasks.set_transfer_buffers, buffers)
        )


def construct_actor_workers(
    config: Config,
    env_class: Union[Type[Env], Type[RLEnv]],
    mp_ctx: BaseContext,
    policy_worker_queue: faster_fifo.Queue,
    report_queue: faster_fifo.Queue,
    workers_ignore_signals: bool = False,
):
    num_environments = config.NUM_ENVIRONMENTS
    actor_worker_queues = [
        faster_fifo.Queue(max_size_bytes=8 * 1024 * 1024)
        for _ in range(num_environments)
    ]

    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    if "*" in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)

    if num_environments > 1:
        if len(scenes) == 0:
            raise RuntimeError(
                "No scenes to load, multiple process logic relies on being able to split scenes uniquely between processes"
            )

        if len(scenes) < num_environments:
            raise RuntimeError(
                "reduce the number of environments as there "
                "aren't enough number of scenes.\n"
                "num_environments: {}\tnum_scenes: {}".format(
                    num_environments, len(scenes)
                )
            )

        random.shuffle(scenes)

    scene_splits: List[List[str]] = [[] for _ in range(num_environments)]
    for idx, scene in enumerate(scenes):
        scene_splits[idx % len(scene_splits)].append(scene)

    assert sum(map(len, scene_splits)) == len(scenes)

    workers = []
    for i in range(num_environments):
        proc_config = config.clone()
        proc_config.defrost()

        task_config = proc_config.TASK_CONFIG
        task_config.SEED = task_config.SEED + i
        if len(scenes) > 0:
            task_config.DATASET.CONTENT_SCENES = scene_splits[i]

        task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = (
            config.SIMULATOR_GPU_ID
        )

        task_config.SIMULATOR.AGENT_0.SENSORS = config.SENSORS

        proc_config.freeze()

        w = ActorWorker(
            mp_ctx,
            i,
            make_env_fn,
            (proc_config, env_class),
            policy_worker_queue,
            report_queue,
            actor_worker_queues[i],
        )
        workers.append(w)

    [w.start() for w in workers]

    return workers, actor_worker_queues
