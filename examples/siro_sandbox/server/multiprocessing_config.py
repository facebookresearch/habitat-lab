

# multiprocessing.dummy uses multithreading instead of multiprocessing
use_dummy = True

if use_dummy:
    from multiprocessing.dummy import Process, Queue, Semaphore
else:
    from multiprocessing import Process, Queue, Semaphore
