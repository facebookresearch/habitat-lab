from typing import Any

# multiprocessing.dummy uses multithreading instead of multiprocessing
use_dummy = False

# User code should import this file and use e.g. multiprocessing_config.Process. Thus
# user code doesn't have to know whether we're using multiprocessing.dummy or
# multiprocessing.

Process: Any
Queue: Any
Semaphore: Any

if use_dummy:
    from multiprocessing.dummy import Process, Queue, Semaphore  # noqa: 0
else:
    from multiprocessing import Process, Queue, Semaphore  # noqa: 0
