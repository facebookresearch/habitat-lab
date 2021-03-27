import functools
import os
import signal
import subprocess
import threading
from os import path as osp
from typing import Any, Callable, Optional, Tuple, Union, overload

import ifcfg
import torch
from torch import distributed as distrib

from habitat import Config, logger

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
DEFAULT_MASTER_ADDR = "127.0.0.1"

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


def resume_state_filename(config: Config) -> str:
    fname = RESUME_STATE_BASE_NAME

    if is_slurm_job() and config.RL.preemption.append_slurm_job_id:
        fname += "-{}".format(SLURM_JOBID)

    return osp.join(config.CHECKPOINT_FOLDER, fname + ".pth")


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
def save_resume_state(state: Any, filename_or_config: Union[Config, str]):
    r"""Saves the resume job state to the specified filename.
        This is useful when working with preemptable job partitions.

    :param state: The state to save
    :param filename_or_config: The filename of the saved state or the config to construct it.
    """
    if isinstance(filename_or_config, Config):
        filename = resume_state_filename(filename_or_config)
    else:
        filename = filename_or_config

    torch.save(state, filename)


def load_resume_state(filename_or_config: Union[Config, str]) -> Optional[Any]:
    r"""Loads the saved resume state

    :param filename_or_config: The filename of the saved state or the config to construct it.

    :return: The saved state if the file exists, else none
    """
    if isinstance(filename_or_config, Config):
        filename = resume_state_filename(filename_or_config)
    else:
        filename = filename_or_config

    if not osp.exists(filename):
        return None

    if rank0_only():
        logger.info(f"Loading resume state: {filename}")

    return torch.load(filename, map_location="cpu")


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

    master_port = int(os.environ.get("MASTER_PORT", DEFAULT_PORT))
    if SLURM_JOBID is not None:
        master_port += int(SLURM_JOBID) % int(
            os.environ.get("MASTER_PORT_RANGE", DEFAULT_PORT_RANGE)
        )
    master_addr = os.environ.get("MASTER_ADDR", DEFAULT_MASTER_ADDR)

    tcp_store = distrib.TCPStore(  # type: ignore
        master_addr, master_port, world_size, world_rank == 0
    )
    distrib.init_process_group(
        backend, store=tcp_store, rank=world_rank, world_size=world_size
    )

    return local_rank, tcp_store
