import queue

from .multiprocessing_config import Queue, Semaphore


class InterprocessRecord:
    def __init__(self, max_steps_ahead):
        self.keyframe_queue = Queue()
        self.client_state_queue = Queue()
        self.step_semaphore = Semaphore(max_steps_ahead)


# How many frames we can simulate "ahead" of what keyframes have been sent.
# A larger value increases lag on the client, while ensuring a more reliable
# simulation rate in the presence of unreliable network comms.
max_steps_ahead = 3

interprocess_record = InterprocessRecord(max_steps_ahead)


def send_keyframe_to_networking_thread(keyframe):
    # Acquire the semaphore to ensure the simulation doesn't advance too far ahead
    interprocess_record.step_semaphore.acquire()

    interprocess_record.keyframe_queue.put(keyframe)


def send_client_state_to_main_thread(client_state):
    interprocess_record.client_state_queue.put(client_state)


def get_queued_keyframes():
    keyframes = []

    while True:
        try:
            keyframe = interprocess_record.keyframe_queue.get(block=False)
            keyframes.append(keyframe)
            interprocess_record.step_semaphore.release()
        except queue.Empty:
            # No more keyframes in the queue, break out of the loop
            break

    return keyframes


def get_single_queued_keyframe():
    try:
        keyframe = interprocess_record.keyframe_queue.get(block=False)
        interprocess_record.step_semaphore.release()
        return keyframe
    except queue.Empty:
        # No keyframes in the queue
        return None


def get_queued_client_states():
    client_states = []

    while True:
        try:
            client_state = interprocess_record.client_state_queue.get(
                block=False
            )
            client_states.append(client_state)
        except queue.Empty:
            # No more keyframes in the queue, break out of the loop
            break

    return client_states
