import abc
import signal
from multiprocessing import Event, Process, context
from multiprocessing.context import BaseContext
from typing import Type

import faster_fifo
import torch

_ForkingPickler = context.reduction.ForkingPickler


def rebuild_queue(newstate, message_buffer_size):
    q = faster_fifo.Queue.__new__(faster_fifo.Queue)
    q.__dict__.update(newstate)
    q.reallocate_msg_buffer(message_buffer_size)
    return q


def reduce_queue(q):
    state = q.__dict__.copy()
    message_buffer_size = (
        0 if q.message_buffer is None else len(q.message_buffer)
    )
    state["message_buffer"] = None
    state["message_buffer_memview"] = None
    return rebuild_queue, (state, message_buffer_size)


_ForkingPickler.register(faster_fifo.Queue, reduce_queue)


class ProcessBase(abc.ABC):
    @abc.abstractmethod
    def run(self):
        pass


class WorkerBase:
    _proc_done_event: Event
    _proc: Process

    def __init__(
        self,
        mp_ctx: BaseContext,
        process_class: Type[ProcessBase],
        *process_args
    ):
        self._proc = None
        self._proc_done_event = mp_ctx.Event()
        self._proc_done_event.clear()
        p = mp_ctx.Process(
            target=self._worker_fn,
            args=(process_class, *process_args, self._proc_done_event),
        )

        p.deamon = True
        p.start()
        self._proc = p

    @staticmethod
    def _worker_fn(process_class: Type[ProcessBase], *process_args):
        #  signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
        signal.signal(signal.SIGUSR2, signal.SIG_IGN)

        torch.set_num_threads(1)

        with torch.no_grad():
            try:
                process_class(*process_args).run()
            except KeyboardInterrupt:
                pass

    def close(self):
        if self._proc is not None:
            self._proc_done_event.set()

    def join(self):
        if self._proc is not None:
            self._proc_done_event.set()
            self._proc.join()
            self._proc = None

    def __del__(self):
        self.join()
