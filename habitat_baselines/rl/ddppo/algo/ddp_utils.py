import os
import os.path as osp
import shlex
import signal
import subprocess
import threading

import ifcfg
import torch
import torch.distributed as distrib
import torch.nn as nn

from habitat import logger

EXIT = threading.Event()
EXIT.clear()
REQUEUE = threading.Event()
REQUEUE.clear()

SLURM_JOBID = os.environ.get("SLURM_JOB_ID", None)
INTERRUPTED_STATE_FILE = osp.join(
    os.environ["HOME"], ".interrupted_states", "{}.pth".format(SLURM_JOBID)
)


def _clean_exit_handler(signum, frame):
    EXIT.set()
    print("Exiting cleanly", flush=True)


def _requeue_handler(signal, frame):
    EXIT.set()
    REQUEUE.set()


def add_signal_handlers():
    signal.signal(signal.SIGINT, _clean_exit_handler)
    signal.signal(signal.SIGTERM, _clean_exit_handler)
    signal.signal(signal.SIGUSR2, _clean_exit_handler)

    signal.signal(signal.SIGUSR1, _requeue_handler)


def save_interrupted_state(state):
    if SLURM_JOBID is None:
        logger.warn("SLURM_JOBID is none, not saving interrupted state")
        return

    torch.save(state, INTERRUPTED_STATE_FILE)


def load_interrupted_state():
    if SLURM_JOBID is None:
        return None

    if not osp.exists(INTERRUPTED_STATE_FILE):
        return None

    return torch.load(INTERRUPTED_STATE_FILE, map_location="cpu")


def requeue_job():
    if SLURM_JOBID is None:
        return

    if not REQUEUE.is_set():
        return

    distrib.barrier()

    if distrib.get_rank() == 0:
        logger.info("Requeueing job {}".format(SLURM_JOBID))
        subprocess.check_call(
            shlex.split("scontrol requeue {}".format(SLURM_JOBID))
        )


def get_ifname():
    return ifcfg.default_interface()["device"]


def init_distrib_slurm(backend="nccl"):
    assert torch.distributed.is_available(), "torch.distributed not avaliable"

    if "GLOO_SOCKET_IFNAME" not in os.environ:
        os.environ["GLOO_SOCKET_IFNAME"] = get_ifname()

    if "NCCL_SOCKET_IFNAME" not in os.environ:
        os.environ["NCCL_SOCKET_IFNAME"] = get_ifname()

    master_port = int(os.environ.get("MASTER_PORT", 8738))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    local_rank = int(
        os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0))
    )
    world_rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world_size = int(
        os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1))
    )

    tcp_store = distrib.TCPStore(
        master_addr, master_port, world_size, world_rank == 0
    )
    distrib.init_process_group(
        backend, store=tcp_store, rank=world_rank, world_size=world_size
    )

    return local_rank, tcp_store
