#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import queue

from habitat_hitl._internal.networking.multiprocessing_config import (
    Queue,
    Semaphore,
)


class InterprocessRecord:
    def __init__(self, max_steps_ahead: int):
        self.keyframe_queue = Queue()
        self.client_state_queue = Queue()
        self.step_semaphore = Semaphore(max_steps_ahead)

    def send_keyframe_to_networking_thread(self, keyframe):
        # Acquire the semaphore to ensure the simulation doesn't advance too far ahead
        self.step_semaphore.acquire()

        self.keyframe_queue.put(keyframe)

    def send_client_state_to_main_thread(self, client_state):
        self.client_state_queue.put(client_state)

    def get_queued_keyframes(self):
        keyframes = []

        while True:
            try:
                keyframe = self.keyframe_queue.get(block=False)
                keyframes.append(keyframe)
                self.step_semaphore.release()
            except queue.Empty:
                # No more keyframes in the queue, break out of the loop
                break

        return keyframes

    def get_single_queued_keyframe(self):
        try:
            keyframe = self.keyframe_queue.get(block=False)
            self.step_semaphore.release()
            return keyframe
        except queue.Empty:
            # No keyframes in the queue
            return None

    def get_queued_client_states(self):
        client_states = []

        while True:
            try:
                client_state = self.client_state_queue.get(block=False)
                client_states.append(client_state)
            except queue.Empty:
                # No more keyframes in the queue, break out of the loop
                break

        return client_states
