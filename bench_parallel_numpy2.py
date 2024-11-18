import time
import threading
import multiprocessing
import numpy as np
import sys
import os
import platform
import argparse
import matplotlib.pyplot as plt

def numpy_worker(barrier, ARRAY_LENGTH, NUM_ITERATIONS):
    """Thread worker function for NumPy computations."""
    x = np.arange(ARRAY_LENGTH, dtype=np.float64)
    barrier.wait()  # Synchronize start
    for _ in range(NUM_ITERATIONS):
        x = x * 1.01  # element-wise operation
        x[0] += x.mean() * 0.01 # reduction operation

def list_worker(barrier, ARRAY_LENGTH, NUM_ITERATIONS):
    """Thread worker function for list computations."""
    x = [float(xi) for xi in range(ARRAY_LENGTH)]
    barrier.wait()  # Synchronize start
    for _ in range(NUM_ITERATIONS):
        x = [xi * 1.01 for xi in x]
        x[0] += sum(x) / len(x) * 0.01

def numpy_process_worker(start_event, ready_counter, lock, ARRAY_LENGTH, NUM_ITERATIONS):
    """Process worker function for NumPy computations."""
    x = np.arange(ARRAY_LENGTH, dtype=np.float64)
    with lock:
        ready_counter.value += 1
    start_event.wait()  # Synchronize start
    for _ in range(NUM_ITERATIONS):
        x = x * 1.01  # element-wise operation
        x[0] += x.mean() * 0.01 # reduction operation

def list_process_worker(start_event, ready_counter, lock, ARRAY_LENGTH, NUM_ITERATIONS):
    """Process worker function for list computations."""
    x = [float(xi) for xi in range(ARRAY_LENGTH)]
    with lock:
        ready_counter.value += 1
    start_event.wait()  # Synchronize start
    for _ in range(NUM_ITERATIONS):
        x = [xi * 1.01 for xi in x]
        x[0] += sum(x) / len(x) * 0.01

def run_benchmark(kind, method, num_workers, ARRAY_LENGTH, NUM_ITERATIONS):
    """
    Runs the benchmark for the specified configuration.

    Args:
        kind (str): 'numpy' or 'list' to specify the data structure.
        method (str): 'thread' or 'process' to specify concurrency method.
        num_workers (int): Number of threads or processes.
    """
    if kind == 'numpy':
        if method == 'threads':
            worker = numpy_worker
        else:
            worker = numpy_process_worker
    elif kind == 'list':
        if method == 'threads':
            worker = list_worker
        else:
            worker = list_process_worker
    else:
        raise ValueError("Unknown kind")

    if method == 'threads':
        # Multithreaded execution
        barrier = threading.Barrier(num_workers + 1)
        threads = []
        for _ in range(num_workers):
            t = threading.Thread(target=worker, args=(barrier, ARRAY_LENGTH, NUM_ITERATIONS))
            threads.append(t)
            t.start()
        barrier.wait()  # Synchronize start
        start_time = time.time()
        # All threads proceed after the barrier is lifted
        for t in threads:
            t.join()
        end_time = time.time()
    elif method == 'processes':
        # Multiprocess execution
        start_event = multiprocessing.Event()
        ready_counter = multiprocessing.Value('i', 0)
        lock = multiprocessing.Lock()
        processes = []
        for _ in range(num_workers):
            p = multiprocessing.Process(target=worker, args=(start_event, ready_counter, lock, ARRAY_LENGTH, NUM_ITERATIONS))
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
    mflop_per_sec = total_ops / (elapsed_time * 1e6)

    return elapsed_time, mflop_per_sec

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark multithreading and multiprocessing with NumPy and list operations.')
    parser.add_argument('--array-length', type=int, default=1000, help='Length of the arrays/lists used in computations.')
    parser.add_argument('--num-iterations', type=int, default=4000, help='Number of iterations each worker performs.')
    parser.add_argument('--num-workers-list', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128], help='List of worker counts to benchmark.')

    args = parser.parse_args()

    ARRAY_LENGTH = args.array_length
    NUM_ITERATIONS = args.num_iterations
    NUM_WORKERS_LIST = args.num_workers_list

    print(f"Python Version: {sys.version}")
    print(f"NumPy Version: {np.__version__}")
    print(f"os.cpu_count(): {os.cpu_count()}")

    # Collect results in a dictionary for table construction
    results = {num_workers: {} for num_workers in NUM_WORKERS_LIST}

    # Initialize plot data
    plot_data = {
        'numpy_processes': {'num_workers': [], 'mflop_per_sec': []},
        'numpy_threads': {'num_workers': [], 'mflop_per_sec': []},
        'list_processes': {'num_workers': [], 'mflop_per_sec': []},
        'list_threads': {'num_workers': [], 'mflop_per_sec': []}
    }

    for num_workers in NUM_WORKERS_LIST:
        for kind in ['numpy', 'list']:
            for method in ['threads', 'processes']:
                elapsed_time, mflop_per_sec = run_benchmark(kind, method, num_workers, ARRAY_LENGTH, NUM_ITERATIONS)
                key = f"{kind}_{method}"
                results[num_workers][key] = (elapsed_time, mflop_per_sec)
                # Update plot data
                plot_data[key]['num_workers'].append(num_workers)
                plot_data[key]['mflop_per_sec'].append(mflop_per_sec)

    # Print results in ASCII table format
    headers = ["# Workers", "Numpy Threads", "Numpy Processes", "List Threads", "List Processes"]
    col_widths = [len(header) for header in headers]

    # Calculate column widths dynamically
    for num_workers in NUM_WORKERS_LIST:
        for key, (elapsed, mflop) in results[num_workers].items():
            idx = ["numpy_threads", "numpy_processes", "list_threads", "list_processes"].index(key)
            content_length = len(f"{elapsed:.2f}s, {mflop:.2f} MFLOPS")
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
        for key in ["numpy_threads", "numpy_processes", "list_threads", "list_processes"]:
            if key in results[num_workers]:
                elapsed, mflop = results[num_workers][key]
                row.append(f"{elapsed:.2f}s, {mflop:.2f} MFLOPS")
            else:
                row.append("N/A")
        print(row_format.format(*row))

    # Generate and save the plot
    plt.figure(figsize=(6.5, 4), tight_layout=True)  # ~35% smaller with tight layout for padding

    # Define styles for the lines
    styles = {
        'numpy_threads': ('solid', '#FFA500'),  # Prettier orange
        'numpy_processes': ('dotted', '#FFA500'),  # Prettier orange
        'list_threads': ('solid', '#1E90FF'),  # Prettier blue
        'list_processes': ('dotted', '#1E90FF'),  # Prettier blue
    }

    for key in plot_data:
        x = plot_data[key]['num_workers']
        y = plot_data[key]['mflop_per_sec']
        linestyle, color = styles[key]
        plt.plot(x, y, linestyle=linestyle, color=color, marker='o', label=key.replace('_', ' ').title(), linewidth=2)  # Thicker lines
            
    # Adjust the labels for the far-right datapoint
    printed_labels = []

    # Add padding to ensure labels outside the plot fit within the figure
    plt.gcf().subplots_adjust(right=0.8)  # Increase right-side padding

    for key in plot_data:
        x = plot_data[key]['num_workers']
        y = plot_data[key]['mflop_per_sec']
        linestyle, color = styles[key]

        # Check if the current value is sufficiently distinct from previously printed labels
        if all(abs(y[-1] - prev) / max(y[-1], prev) > 0.2 for prev in printed_labels):
            # Position the label just outside the right edge of the plot
            x_pos = plt.xlim()[1] * 1.25  # 105% of the x-axis range (outside the plot)
            plt.text(
                x_pos, y[-1], f"{y[-1]:.1f}",
                fontsize=12, color=color, ha='left', va='center'
            )
            printed_labels.append(y[-1])  # Mark this label as printed


    plt.xlabel('Number of Workers', fontsize=14)  # Larger text
    plt.ylabel('MFLOPS', fontsize=14)  # Larger text
    plt.title(f'MFLOPS for array length {ARRAY_LENGTH}', fontsize=16)  # Larger title
    plt.legend(fontsize=12)  # Larger legend text
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xticks(NUM_WORKERS_LIST, labels=NUM_WORKERS_LIST, fontsize=12)  # Larger x-axis labels
    plt.yticks(fontsize=12)  # Larger y-axis labels
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    # Add padding to ensure text doesn't get cut off
    plt.gcf().subplots_adjust(right=0.9, bottom=0.2)  # Extra padding

    filename = f'mflops_array_length_{ARRAY_LENGTH}.png'
    plt.savefig(filename)
    print(f'Plot saved to {filename}')


