import time
import threading
import multiprocessing
import numpy as np
import sys
import os
import argparse
import matplotlib.pyplot as plt

def numpy_worker(sync_func, array_length, num_iterations):
    """Worker function for NumPy computations."""
    x = np.arange(array_length, dtype=np.float64)
    sync_func()  # Synchronize start
    for _ in range(num_iterations):
        x = x * 1.01  # Element-wise operation
        x[0] += x.mean() * 0.01  # Reduction operation

def list_worker(sync_func, array_length, num_iterations):
    """Worker function for list computations."""
    x = [float(xi) for xi in range(array_length)]
    sync_func()  # Synchronize start
    for _ in range(num_iterations):
        x = [xi * 1.01 for xi in x]  # Element-wise operation
        x[0] += sum(x) / len(x) * 0.01  # Reduction operation

def launch_workers(worker_func, num_workers, method, array_length, num_iterations):
    """Launches workers using threading or multiprocessing."""
    if method == 'threads':
        barrier = threading.Barrier(num_workers + 1)
        def sync_func():
            barrier.wait()
        workers = []
        for _ in range(num_workers):
            t = threading.Thread(target=worker_func, args=(sync_func, array_length, num_iterations))
            workers.append(t)
            t.start()
        barrier.wait()  # Synchronize all threads
        start_time = time.time()
        for t in workers:
            t.join()
        end_time = time.time()
    elif method == 'processes':
        # Use a multiprocessing.Event for synchronization
        start_event = multiprocessing.Event()
        def sync_func():
            start_event.wait()
        workers = []
        for _ in range(num_workers):
            p = multiprocessing.Process(target=worker_func, args=(sync_func, array_length, num_iterations))
            workers.append(p)
            p.start()
        start_time = time.time()
        start_event.set()  # Signal all processes to start
        for p in workers:
            p.join()
        end_time = time.time()
    else:
        raise ValueError("Unknown method")
    return start_time, end_time

def run_benchmark(kind, method, num_workers, args):
    """Runs the benchmark for the specified configuration."""
    if kind == 'numpy':
        worker = numpy_worker
    elif kind == 'list':
        worker = list_worker
    else:
        raise ValueError("Unknown kind")

    # Start workers and measure time
    start_time, end_time = launch_workers(worker, num_workers, method, args.array_length, args.num_iterations)

    elapsed_time = end_time - start_time
    total_ops = num_workers * args.num_iterations * args.array_length * 2
    mflop_per_sec = total_ops / (elapsed_time * 1e6)
    return elapsed_time, mflop_per_sec

def print_results_table(results, num_workers_list):
    """Prints the benchmark results in a formatted table."""
    headers = ["# Workers", "Numpy Threads", "Numpy Processes", "List Threads", "List Processes"]
    row_format = "{:<10} | {:<23} | {:<23} | {:<23} | {:<23}"
    separator = "-" * 10 + "-+-" + "-+-".join(["-" * 23] * 4)

    print()
    print(row_format.format(*headers))
    print(separator)
    for num_workers in num_workers_list:
        row = [str(num_workers)]
        for key in ["numpy_threads", "numpy_processes", "list_threads", "list_processes"]:
            if key in results[num_workers]:
                elapsed, mflop = results[num_workers][key]
                row.append(f"{elapsed:.2f}s, {mflop:.2f} MFLOPS")
            else:
                row.append("N/A")
        print(row_format.format(*row))

def save_plot(plot_data, array_length):
    """Generates and saves the benchmark plot."""
    plt.figure(figsize=(6.5, 4), tight_layout=True)
    styles = {
        'numpy_threads': ('solid', '#FFA500'),
        'numpy_processes': ('dotted', '#FFA500'),
        'list_threads': ('solid', '#1E90FF'),
        'list_processes': ('dotted', '#1E90FF'),
    }

    for key in plot_data:
        x = plot_data[key]['num_workers']
        y = plot_data[key]['mflop_per_sec']
        linestyle, color = styles[key]
        plt.plot(x, y, linestyle=linestyle, color=color, marker='o', label=key.replace('_', ' ').title(), linewidth=2)

    # add labels for the far-right datapoints
    printed_labels = []
    # Add padding to ensure labels outside the plot fit within the figure
    plt.gcf().subplots_adjust(right=0.8)
    for key in plot_data:
        x = plot_data[key]['num_workers']
        y = plot_data[key]['mflop_per_sec']
        _, color = styles[key]

        # Check if the current value is sufficiently distinct from previously printed labels
        if all(abs(y[-1] - prev) / max(y[-1], prev) > 0.2 for prev in printed_labels):
            # Position the label just outside the right edge of the plot
            x_pos = plt.xlim()[1] * 1.25  # 125% of the x-axis range (outside the plot)
            plt.text(
                x_pos, y[-1], f"{y[-1]:.1f}",
                fontsize=12, color=color, ha='left', va='center'
            )
            printed_labels.append(y[-1])  # Mark this label as printed
            
    plt.xlabel('Number of Workers', fontsize=14)
    plt.ylabel('MFLOPS', fontsize=14)
    plt.title(f'MFLOPS for array length {array_length}', fontsize=16)
    plt.legend(fontsize=12)
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xticks(plot_data['numpy_threads']['num_workers'], labels=plot_data['numpy_threads']['num_workers'], fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.gcf().subplots_adjust(right=0.9, bottom=0.2)
    filename = f'mflops_array_length_{array_length}.png'
    plt.savefig(filename)
    print(f'Plot saved to {filename}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark multithreading and multiprocessing with NumPy and list operations.')
    parser.add_argument('--array-length', type=int, default=1000, help='Length of the arrays/lists used in computations.')
    parser.add_argument('--num-iterations', type=int, default=4000, help='Number of iterations each worker performs.')
    parser.add_argument('--num-workers-list', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128], help='List of worker counts to benchmark.')
    args = parser.parse_args()

    print(f"Python Version: {sys.version}")
    print(f"NumPy Version: {np.__version__}")
    print(f"os.cpu_count(): {os.cpu_count()}")

    results = {num_workers: {} for num_workers in args.num_workers_list}
    plot_data = {
        'numpy_threads': {'num_workers': [], 'mflop_per_sec': []},
        'numpy_processes': {'num_workers': [], 'mflop_per_sec': []},
        'list_threads': {'num_workers': [], 'mflop_per_sec': []},
        'list_processes': {'num_workers': [], 'mflop_per_sec': []}
    }

    for num_workers in args.num_workers_list:
        for kind in ['numpy', 'list']:
            for method in ['threads', 'processes']:
                elapsed_time, mflop_per_sec = run_benchmark(kind, method, num_workers, args)
                key = f"{kind}_{method}"
                results[num_workers][key] = (elapsed_time, mflop_per_sec)
                plot_data[key]['num_workers'].append(num_workers)
                plot_data[key]['mflop_per_sec'].append(mflop_per_sec)

    print_results_table(results, args.num_workers_list)
    save_plot(plot_data, args.array_length)
