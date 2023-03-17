#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import enum
import inspect
import queue
import signal
from multiprocessing import SimpleQueue
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Barrier, Event, Lock
from typing import Any, Callable, Dict, List, Optional, Type

import attr
import threadpoolctl
import torch

from habitat_baselines.rl.ver.queue import BatchedQueue


@attr.s(auto_attribs=True, init=False, slots=True)
class WorkerQueues:
    environments: List[BatchedQueue]
    inference: BatchedQueue
    report: BatchedQueue
    preemption_decider: BatchedQueue

    def __init__(self, num_environments):
        self.environments = [
            BatchedQueue(8 * 1024 * 1024) for _ in range(num_environments)
        ]
        self.inference = BatchedQueue(1024 * 1024 * num_environments)
        self.report = BatchedQueue(1024 * 1024 * num_environments)
        self.preemption_decider = BatchedQueue(1024 * 1024 * num_environments)


@attr.s(auto_attribs=True, init=False, slots=True)
class RolloutEarlyEnds:
    steps: Synchronized
    time: Synchronized

    def __init__(self, mp_ctx: BaseContext) -> None:
        self.steps = mp_ctx.Value("d", -1.0, lock=False)
        self.time = mp_ctx.Value("d", -1.0, lock=False)


@attr.s(auto_attribs=True, slots=True, init=False)
class InferenceWorkerSync:
    lock: Lock
    all_workers: Barrier
    should_start_next: Event
    rollout_done: Event

    def __init__(
        self,
        mp_ctx: BaseContext,
        n_inference_workers: int,
    ) -> None:
        self.lock = mp_ctx.Lock()
        self.all_workers = mp_ctx.Barrier(n_inference_workers)
        self.should_start_next = mp_ctx.Event()
        self.should_start_next.set()
        self.rollout_done = mp_ctx.Event()
        self.rollout_done.clear()


@attr.s(auto_attribs=True)
class ProcessBase:
    done_event: Event
    response_queue: SimpleQueue
    _dispatch_table: Dict[enum.Enum, Callable[[Any], None]] = attr.ib(
        init=False, factory=dict
    )

    @staticmethod
    def _build_dispatcher(
        func: Callable[..., None], n_params: int, task: enum.Enum
    ) -> Callable[[Any], None]:
        def _dispatcher(data: Optional[Any]) -> None:
            if n_params == 0:
                if data is not None:
                    raise RuntimeError(
                        f"Function for task {task} does not take a data argument, "
                        f"but data\n{data}\n was given."
                    )
                func()
            else:
                if data is None:
                    raise RuntimeError(
                        f"Function for task {task} takes no data."
                    )
                elif n_params == 1:
                    func(data)
                elif len(data) != n_params:
                    raise RuntimeError(
                        f"Function for task {task} takes {n_params} but only got {len(data)}"
                    )
                else:
                    if isinstance(data, dict):
                        func(**data)
                    else:
                        func(*data)

        return _dispatcher

    def build_dispatch_table(self, task_enum: enum.EnumMeta) -> None:
        for task in task_enum:  # type: enum.Enum
            func = getattr(self, task.name.lower())
            assert func is not None

            sig = inspect.signature(func)

            self._dispatch_table[task] = self._build_dispatcher(
                func, len(sig.parameters), task
            )

    def dispatch_task(self, task: enum.Enum, data: Optional[Any]) -> None:
        if task not in self._dispatch_table:
            raise RuntimeError(f"Unknown task: {task}")

        self._dispatch_table[task](data)

    @property
    def task_queue(self) -> BatchedQueue:
        raise NotImplementedError("Must implement the task_queue property")

    def run(self):
        while not self.done_event.is_set():
            try:
                tasks_datas = self.task_queue.get_many(timeout=1.0)
            except queue.Empty:
                pass
            else:
                for task, data in tasks_datas:
                    self.dispatch_task(task, data)


@attr.s(auto_attribs=True, init=False, slots=True)
class WorkerBase:
    _proc_done_event: Event
    response_queue: SimpleQueue
    _proc: Optional[BaseProcess]

    def __init__(
        self,
        mp_ctx: BaseContext,
        process_class: Type[ProcessBase],
        *process_args,
        **process_kwargs,
    ):
        assert issubclass(process_class, ProcessBase)
        self._proc = None

        self._proc_done_event = mp_ctx.Event()
        self._proc_done_event.clear()
        self.response_queue = mp_ctx.SimpleQueue()
        p = mp_ctx.Process(  # type: ignore[attr-defined]
            target=self._worker_fn,
            args=(
                process_class,
                (
                    self._proc_done_event,
                    self.response_queue,
                    *process_args,
                ),
                process_kwargs,
            ),
        )

        p.daemon = True
        p.start()
        self._proc = p

    @staticmethod
    def _worker_fn(
        process_class: Type[ProcessBase], process_args, process_kwargs
    ):
        #  signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
        signal.signal(signal.SIGUSR2, signal.SIG_IGN)

        torch.set_num_threads(1)
        threadpoolctl.threadpool_limits(limits=1)

        with torch.no_grad():
            try:
                process_class(*process_args, **process_kwargs).run()
            except KeyboardInterrupt:
                pass

    def close(self):
        if self._proc is not None:
            self._proc_done_event.set()

    def join(self):
        self.close()

        if self._proc is not None:
            self._proc.join(5.0)
            if self._proc.is_alive():
                self._proc.kill()
            self._proc = None

    def __del__(self):
        self.join()
