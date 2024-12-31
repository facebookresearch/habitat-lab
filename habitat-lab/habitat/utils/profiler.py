import threading
import multiprocessing
import time
from multiprocessing import Manager

# Initialize a Manager to create shared data structures
_manager = Manager()

# Shared list to store profiling events from all processes
_profiling_events = _manager.list()

# Thread-local storage to track profiling context per thread
_profiling_state = threading.local()

# Lock for thread safety within each process
_profiling_lock = threading.Lock()

class ProfilingContext:
    def __init__(self, name):
        self.name = name
        self.thread_id = threading.get_ident()
        self.process_name = multiprocessing.current_process().name

    def __enter__(self):
        # Ensure thread-local variables are initialized for this thread
        if not hasattr(_profiling_state, 'in_context'):
            _profiling_state.in_context = False
            _profiling_state.current_block_name = None

        # Check for nested contexts
        if _profiling_state.in_context:
            raise RuntimeError(
                f"Nested profiling contexts are not allowed. "
                f"Current block '{_profiling_state.current_block_name}' is already active."
            )

        # Set the current context state and block name
        _profiling_state.in_context = True
        _profiling_state.current_block_name = self.name
        self.start_time = time.time()

        # Record the start event
        event = {
            'timestamp': self.start_time,
            'thread_id': self.thread_id,
            'process_name': self.process_name,
            'event': 'start',
            'name': self.name
        }
        _record_event(event)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        duration_ms = (end_time - self.start_time) * 1000

        # Record the end event
        event = {
            'timestamp': end_time,
            'thread_id': self.thread_id,
            'process_name': self.process_name,
            'event': 'end',
            'name': self.name,
            'duration_ms': duration_ms
        }
        _record_event(event)

        # Reset the profiling context state
        _profiling_state.in_context = False
        _profiling_state.current_block_name = None

def _record_event(event):
    with _profiling_lock:
        _profiling_events.append(event)

def print_profiling_events():
    # Collect events from the shared list
    events = list(_profiling_events)

    if not events:
        print("No profiling events to display.")
        return

    # Collect unique (process_name, thread_id) pairs and assign column indices
    thread_ids = sorted(set(
        (event['process_name'], event['thread_id']) for event in events
    ))
    thread_id_to_column = {tid: idx for idx, tid in enumerate(thread_ids)}
    num_columns = len(thread_ids)

    # Sort events by their timestamp
    sorted_events = sorted(events, key=lambda e: e['timestamp'])

    # Prepare and print the header
    header = '   '.join(
        f"{tid[0]} - thread {idx}".ljust(30) for idx, tid in enumerate(thread_ids)
    )
    print(header)

    # Generate and print each line
    for event in sorted_events:
        line = [''.ljust(30) for _ in range(num_columns)]
        idx = thread_id_to_column[(event['process_name'], event['thread_id'])]
        if event['event'] == 'start':
            line[idx] = event['name'].ljust(30)
        elif event['event'] == 'end':
            line[idx] = f"end {event['name']}, {event['duration_ms']:.3f}ms".ljust(30)
        print('   '.join(line))
