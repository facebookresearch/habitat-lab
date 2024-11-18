import time
import threading
import multiprocessing
import numpy as np
import sys
import os

# Configuration variables
NUM_ITERATIONS = 2000      # Number of iterations each worker performs
ARRAY_LENGTH = 1000     # Length of the arrays/lists used in computations
NUM_WORKERS_LIST = [1, 2, 4, 8, 16, 32, 64, 128]  # List of worker counts to benchmark

def numpy_worker(barrier):
    """Thread worker function for NumPy computations."""
    x = np.arange(ARRAY_LENGTH, dtype=np.float64)
    barrier.wait()  # Synchronize start
    for _ in range(NUM_ITERATIONS):
        x = x * 1.01  # element-wise operation
        x[0] += x.mean() # reduction operation

def list_worker(barrier):
    """Thread worker function for list computations."""
    x = [float(xi) for xi in range(ARRAY_LENGTH)]
    barrier.wait()  # Synchronize start
    for _ in range(NUM_ITERATIONS):
        x = [xi * 1.01 for xi in x]
        x[0] += sum(x) / len(x)

def numpy_process_worker(start_event, ready_counter, lock):
    """Process worker function for NumPy computations."""
    precomputed_data = np.arange(ARRAY_LENGTH, dtype=float)
    with lock:
        ready_counter.value += 1
    start_event.wait()  # Synchronize start
    for _ in range(NUM_ITERATIONS):
        b = precomputed_data * 2.0  # Example operation
        total = b.sum()              # Additional operation

def list_process_worker(start_event, ready_counter, lock):
    """Process worker function for list computations."""
    precomputed_data = [float(x) for x in range(ARRAY_LENGTH)]
    with lock:
        ready_counter.value += 1
    start_event.wait()  # Synchronize start
    for _ in range(NUM_ITERATIONS):
        b = [x * 2.0 for x in precomputed_data]  # Example operation
        total = sum(b)                           # Additional operation

def run_benchmark(kind, method, num_workers):
    """
    Runs the benchmark for the specified configuration.

    Args:
        kind (str): 'numpy' or 'list' to specify the data structure.
        method (str): 'thread' or 'process' to specify concurrency method.
        num_workers (int): Number of threads or processes.
    """
    if kind == 'numpy':
        
        if method == 'thread':
            worker = numpy_worker
        else:
            worker = numpy_process_worker
    elif kind == 'list':
        
        if method == 'thread':
            worker = list_worker
        else:
            worker = list_process_worker
    else:
        raise ValueError("Unknown kind")

    if method == 'thread':
        # Multithreaded execution
        barrier = threading.Barrier(num_workers + 1)
        threads = []
        for _ in range(num_workers):
            t = threading.Thread(target=worker, args=(barrier,))
            threads.append(t)
            t.start()
        barrier.wait()  # Synchronize start
        start_time = time.time()
        # All threads proceed after the barrier is lifted
        for t in threads:
            t.join()
        end_time = time.time()
    elif method == 'process':
        # Multiprocess execution
        start_event = multiprocessing.Event()
        ready_counter = multiprocessing.Value('i', 0)
        lock = multiprocessing.Lock()
        processes = []
        for _ in range(num_workers):
            p = multiprocessing.Process(target=worker, args=(start_event, ready_counter, lock))
            processes.append(p)
            p.start()
        # Wait until all processes are ready
        while True:
            with lock:
                if ready_counter.value >= num_workers:
                    break
            time.sleep(0.01)  # Sleep briefly to prevent busy waiting
        start_time = time.time()
        start_event.set()  # Signal all processes to start computation
        for p in processes:
            p.join()
        end_time = time.time()
    else:
        raise ValueError("Unknown method")

    elapsed_time = end_time - start_time
    # Calculate total operations: num_workers * iterations * elements per iteration * 2 operations
    total_ops = num_workers * NUM_ITERATIONS * ARRAY_LENGTH * 2
    mega_ops_per_sec = total_ops / (elapsed_time * 1e6)

    return elapsed_time, mega_ops_per_sec

if __name__ == '__main__':

    print(f"Python Version: {sys.version}")
    print(f"NumPy Version: {np.__version__}")
    print(f"ARRAY_LENGTH: {ARRAY_LENGTH}")
    print(f"NUM_ITERATIONS: {NUM_ITERATIONS}")

    # Collect results in a dictionary for table construction
    results = {num_workers: {} for num_workers in NUM_WORKERS_LIST}

    for num_workers in NUM_WORKERS_LIST:
        for kind in ['numpy', 'list']:
            for method in ['thread', 'process']:
                elapsed_time, mega_ops_per_sec = run_benchmark(kind, method, num_workers)
                results[num_workers][f"{kind}_{method}"] = (elapsed_time, mega_ops_per_sec)

    # Print results in ASCII table format
    headers = ["# Workers", "Numpy Threads", "Numpy Processes", "List Threads", "List Processes"]
    col_widths = [len(header) for header in headers]

    # Calculate column widths dynamically
    for num_workers in NUM_WORKERS_LIST:
        for key, (elapsed, mega_ops) in results[num_workers].items():
            idx = ["numpy_thread", "numpy_process", "list_thread", "list_process"].index(key)
            content_length = len(f"{elapsed:.2f}s, {mega_ops:.2f} Mop/s")
            col_widths[idx + 1] = max(col_widths[idx + 1], content_length)

    # Format strings
    row_format = " | ".join(f"{{:<{width}}}" for width in col_widths)

    # Print header
    print()
    print(row_format.format(*headers))
    print("-+-".join('-' * width for width in col_widths))

    # Print rows
    for num_workers in NUM_WORKERS_LIST:
        row = [str(num_workers)]
        for key in ["numpy_thread", "numpy_process", "list_thread", "list_process"]:
            if key in results[num_workers]:
                elapsed, mega_ops = results[num_workers][key]
                row.append(f"{elapsed:.2f}s, {mega_ops:.2f} Mop/s")
            else:
                row.append("N/A")
        print(row_format.format(*row))
