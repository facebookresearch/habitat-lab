# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import functools
import io
import os
import pickle
import signal
import socket
import subprocess
import threading
from os import path as osp
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import ifcfg
import numpy as np
import torch
from omegaconf import DictConfig
from torch import distributed as distrib

from habitat import logger

T = TypeVar("T")

EXIT = threading.Event()
EXIT.clear()
REQUEUE = threading.Event()
REQUEUE.clear()
SAVE_STATE = threading.Event()
SAVE_STATE.clear()


# Default port to initialized the TCP store on
DEFAULT_PORT = 8738
DEFAULT_PORT_RANGE = 127
# Default address of world rank 0
DEFAULT_MAIN_ADDR = "127.0.0.1"

SLURM_JOBID = os.environ.get("SLURM_JOB_ID", None)
RESUME_STATE_BASE_NAME = ".habitat-resume-state"


def is_slurm_job() -> bool:
    return SLURM_JOBID is not None


def is_slurm_batch_job() -> bool:
    r"""Heuristic to determine if a slurm job is a batch job or not. Batch jobs
    will have a job name that is not a shell unless the user specifically set the job
    name to that of a shell. Interactive jobs have a shell name as their job name.
    """
    return is_slurm_job() and os.environ.get("SLURM_JOB_NAME", None) not in (
        None,
        "bash",
        "zsh",
        "fish",
        "tcsh",
        "sh",
        "interactive",
    )


def resume_state_filename(config: DictConfig, filename_key: str = "") -> str:
    fname = RESUME_STATE_BASE_NAME

    if (
        is_slurm_job()
        and config.habitat_baselines.rl.preemption.append_slurm_job_id
    ):
        fname += "-{}".format(SLURM_JOBID)

    return (
        osp.join(config.habitat_baselines.checkpoint_folder, fname)
        + filename_key
        + ".pth"
    )


@overload
def rank0_only() -> bool:
    ...


@overload
def rank0_only(fn: Callable) -> Callable:
    ...


def rank0_only(fn: Optional[Callable] = None) -> Union[Callable, bool]:
    r"""Helper function to only execute code if a process is world rank 0

    Can be used both as a function in an if statement,

    .. code:: py

        if rank0_only():
            ...

    or as a decorator,

    .. code:: py

        @rank0_only
        def fn_for_r0_only(...):
            ...

    :param fn: Function to wrap and only execute if the process is rank 0.
        If a process is rank 0, the function will be run and it's return value
        will be returned.  If a process is not rank 0, then the function will not
        be ran and :py:`None` will be returned.

    :return: The wrapped function if :p:`fn` is not :py:`None`, otherwise
        whether or not this process is rank 0
    """
    if fn is None:
        return (
            not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )

    @functools.wraps(fn)
    def _wrapper(*args, **kwargs):
        if rank0_only():
            return fn(*args, **kwargs)
        return None

    return _wrapper


def _ignore_handler(signum, frame):
    pass


def _clean_exit_handler(signum, frame):
    EXIT.set()
    print("Exiting cleanly", flush=True)


def _clean_exit_and_save_handler(signum, frame):
    EXIT.set()
    SAVE_STATE.set()
    print("Exiting cleanly and saving state", flush=True)


def _requeue_handler(signal, frame):
    REQUEUE.set()
    SAVE_STATE.set()
    EXIT.set()
    print("Got signal to requeue", flush=True)


def add_signal_handlers() -> None:
    signal.signal(signal.SIGCONT, _ignore_handler)
    signal.signal(signal.SIGINT, _clean_exit_handler)

    # SIGUSR2 can be sent to all processes to have them cleanup
    # and exit nicely.  This is nice to use with SLURM as scancel <job_id>
    # sets a 30 second timer for the job to exit, and it can take more than
    # 30 seconds for the job to cleanup and exit nicely.  When using NCCL,
    # forcing the job to exit without cleaning up can be bad.
    # scancel --signal SIGUSR2 <job_id> will set no such timer and will give
    # the job ample time to cleanup and exit.
    signal.signal(signal.SIGUSR2, _clean_exit_handler)

    # SLURM always sends SIGTERM so we can use this to save and exit
    signal.signal(signal.SIGTERM, _clean_exit_and_save_handler)

    signal.signal(signal.SIGUSR1, _requeue_handler)


@rank0_only
def save_resume_state(
    state: Any,
    filename_or_config: Union[DictConfig, str],
    filename_key: str = "",
):
    r"""Saves the resume job state to the specified filename.
        This is useful when working with preemptable job partitions.

    :param state: The state to save
    :param filename_or_config: The filename of the saved state or the config to construct it.
    :param filename_key: If generating the filename from the config, append this to the name.
    """
    if isinstance(filename_or_config, DictConfig):
        filename = resume_state_filename(filename_or_config, filename_key)
    else:
        filename = filename_or_config

    torch.save(state, filename)


def load_resume_state(
    filename_or_config: Union[DictConfig, str], filename_key: str = ""
) -> Optional[Any]:
    r"""Loads the saved resume state

    :param filename_or_config: The filename of the saved state or the config to construct it.
    :param filename_key: If generating the filename from the config, append this to the name.

    :return: The saved state if the file exists, else none
    """
    if isinstance(filename_or_config, DictConfig):
        filename = resume_state_filename(filename_or_config, filename_key)
    else:
        filename = filename_or_config

    if not osp.exists(filename):
        return None

    if rank0_only():
        logger.info(f"Loading resume state: {filename}")

    return torch.load(filename, map_location="cpu", weights_only=False)


def requeue_job():
    r"""Requeues the job by calling ``scontrol requeue ${SLURM_JOBID}``"""
    if not is_slurm_batch_job():
        return

    if not REQUEUE.is_set():
        return

    if distrib.is_initialized():
        distrib.barrier()

    if rank0_only():
        logger.info(f"Requeueing job {SLURM_JOBID}")
        subprocess.check_call(["scontrol", "requeue", str(SLURM_JOBID)])


def get_ifname() -> str:
    return ifcfg.default_interface()["device"]


def get_distrib_size() -> Tuple[int, int, int]:
    # Check to see if we should parse from torch.distributed.launch
    if os.environ.get("LOCAL_RANK", None) is not None:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    # Else parse from SLURM is using SLURM
    elif os.environ.get("SLURM_JOBID", None) is not None:
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
    # Otherwise setup for just 1 process, this is nice for testing
    else:
        local_rank = 0
        world_rank = 0
        world_size = 1

    return local_rank, world_rank, world_size


def get_main_addr() -> str:
    return os.environ.get("MAIN_ADDR", DEFAULT_MAIN_ADDR)


def init_distrib_slurm(
    backend: str = "nccl",
) -> Tuple[int, torch.distributed.TCPStore]:  # type: ignore
    r"""Initializes torch.distributed by parsing environment variables set
        by SLURM when ``srun`` is used or by parsing environment variables set
        by torch.distributed.launch

    :param backend: Which torch.distributed backend to use

    :returns: Tuple of the local_rank (aka which GPU to use for this process)
        and the TCPStore used for the rendezvous
    """
    assert (
        torch.distributed.is_available()
    ), "torch.distributed must be available"

    if "GLOO_SOCKET_IFNAME" not in os.environ:
        os.environ["GLOO_SOCKET_IFNAME"] = get_ifname()

    if "NCCL_SOCKET_IFNAME" not in os.environ:
        os.environ["NCCL_SOCKET_IFNAME"] = get_ifname()

    local_rank, world_rank, world_size = get_distrib_size()

    main_port = int(os.environ.get("MAIN_PORT", DEFAULT_PORT))
    if SLURM_JOBID is not None:
        main_port += int(SLURM_JOBID) % int(
            os.environ.get("MAIN_PORT_RANGE", DEFAULT_PORT_RANGE)
        )
    main_addr = get_main_addr()

    tcp_store = distrib.TCPStore(  # type: ignore
        main_addr, main_port, world_size, world_rank == 0
    )
    distrib.init_process_group(
        backend, store=tcp_store, rank=world_rank, world_size=world_size
    )

    return local_rank, tcp_store


def find_free_port() -> int:
    """
    Returns a free port on the system.
    Note that this can only be used to find a port for torch.distribted
    if it's called by a process on the node that will have
    world_rank == 0 and then all ranks are created. If you
    just called `find_free_port()` on each rank independently, every
    rank will have a different port!
    """
    with contextlib.closing(
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("localhost", 0))
        _, port = sock.getsockname()
        return port


def get_free_port_distributed(
    key_name: str, tcp_store: Optional[distrib.TCPStore]
) -> int:
    r"""Return a free port from :py:ref:`find_free_port` and synchronize it across
    all ranks

    :param key_name: The name for this port. This must be unique for each call into this method
        and the same across ranks.
    :param tcp_store: A torch TCPStore that has all ranks. This is used for synchronizing
        the port. Only needed if world_size > 1.
    """
    _port_key = f"_hab_dist_port_{key_name}"
    if rank0_only():
        port = find_free_port()
        if distrib.is_initialized():
            assert tcp_store is not None
            tcp_store.set(_port_key, str(port))
    else:
        assert tcp_store is not None
        tcp_store.wait([_port_key])
        port = int(tcp_store.get(_port_key))

    return port


def _rank_to_relative_rank(
    rank: int, output_rank: int, world_size: int
) -> int:
    return (
        rank - output_rank
        if rank >= output_rank
        else rank - output_rank + world_size
    )


def gatherv(
    t: torch.Tensor, output_rank: int = 0
) -> Optional[List[torch.Tensor]]:
    r"""Distributed gather that works on tensors of variable size.

    Currently on works on tensors with 1 dimension.

    :param t: This rank's tensor to be sent to :ref:`output_rank`
    :param output_rank: The rank the return everyone's inputs to.

    :return: The list of inputs if this rank is :ref:`output_rank`, else :py:`None`.
    """
    assert t.ndim == 1
    assert output_rank >= 0

    world_size = distrib.get_world_size() if distrib.is_initialized() else 1
    if world_size == 1:
        return [t]

    rank = distrib.get_rank()
    is_mine = rank == output_rank

    my_size = torch.tensor(t.numel(), dtype=torch.int64, device=t.device)
    sizes = my_size.view(1).repeat(world_size)

    distrib.all_gather(list(sizes.unbind(0)), my_size)
    sizes = sizes.cpu()
    max_size = sizes.max().item()

    if torch.all(sizes == max_size):
        if is_mine:
            output = list(
                torch.empty(
                    (world_size, max_size), dtype=t.dtype, device=t.device
                ).unbind(0)
            )
        else:
            output = None

        distrib.gather(t, output, output_rank)
    else:
        relative_rank = _rank_to_relative_rank(rank, output_rank, world_size)

        mask = 1
        output = [t]
        while mask < world_size:
            handles = []
            if (relative_rank & mask) == 0:
                src = relative_rank | mask
                if src < world_size:
                    num_msgs = min(mask, world_size - src)

                    src_real = (src + output_rank) % world_size
                    for i in range(num_msgs):
                        output.append(
                            torch.empty(
                                (sizes[(src_real + i) % world_size],),
                                dtype=t.dtype,
                                device=t.device,
                            )
                        )
                        handles.append(
                            torch.distributed.irecv(
                                output[-1], src_real, tag=i
                            )
                        )
            else:
                dst = relative_rank ^ mask
                dst_real = (dst + output_rank) % world_size
                for i, v in enumerate(output):
                    assert v.numel() == sizes[(rank + i) % world_size]
                    handles.append(torch.distributed.isend(v, dst_real, tag=i))

            [h.wait() for h in handles]

            if (relative_rank & mask) != 0:
                output = None
                break

            mask = mask << 1

        if output is not None:
            # We need to re-order the output list to be wrt world ranks
            # instead of relative ranks
            output = [
                output[_rank_to_relative_rank(i, output_rank, world_size)]
                for i in range(world_size)
            ]

    if is_mine:
        assert output is not None
    else:
        assert output is None

    return output


def gather_objects(
    obj: T, device: Optional[torch.device] = None, output_rank: int = 0
) -> Optional[List[T]]:
    r"""Distributed gather on arbitrary python objects. Uses torch.distributed
    under the hood.

    :param obj: This rank's object to be send to :ref:`output_rank`
    :param device: The device to put the tensor that holds the encoded object. Defaults to CPU.
    :param output_rank: The rank to return everyone's inputs to.

    :return: The list of objects if this rank is :ref:`output_rank`, else :py:`None`.
    """
    device = device or torch.device("cpu")
    assert output_rank >= 0

    buf = io.BytesIO()
    pickle.Pickler(buf, protocol=pickle.HIGHEST_PROTOCOL).dump(obj)
    encoded_obj = torch.from_numpy(
        np.frombuffer(buf.getbuffer(), dtype=np.uint8)
    ).to(device=device)
    buf = None

    output = gatherv(
        encoded_obj,
        output_rank=output_rank,
    )
    encoded_obj = None

    if output is not None:
        output = [pickle.loads(bytes(t.cpu())) for t in output]

    return output
