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
    def __init__(self, networking_config, max_steps_ahead: int):
        self.keyframe_queue = Queue()
        self.client_state_queue = Queue()
        self.step_semaphore = Semaphore(max_steps_ahead)
        self.networking_config = networking_config
        self.connection_record_queue = Queue()

    def send_keyframe_to_networking_thread(self, keyframe):
        # Acquire the semaphore to ensure the simulation doesn't advance too far ahead
        self.step_semaphore.acquire()

        self.keyframe_queue.put(keyframe)

    def send_client_state_to_main_thread(self, client_state):
        self.client_state_queue.put(client_state)

    def send_connection_record_to_main_thread(self, connection_record):
        assert "connectionId" in connection_record
        assert "isClientReady" in connection_record
        self.connection_record_queue.put(connection_record)

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

    @staticmethod
    def _dequeue_all(my_queue):
        items = []

        while True:
            try:
                item = my_queue.get(block=False)
                items.append(item)
            except queue.Empty:
                # No more keyframes in the queue, break out of the loop
                break

        return items

    def get_queued_client_states(self):
        return self._dequeue_all(self.client_state_queue)

    def get_queued_connection_records(self):
        return self._dequeue_all(self.connection_record_queue)
