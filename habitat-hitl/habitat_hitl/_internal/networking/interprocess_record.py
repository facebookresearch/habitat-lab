#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from multiprocessing import Queue
from multiprocessing import Semaphore as create_semaphore
from multiprocessing.synchronize import Semaphore
from typing import Any, List, Optional


class InterprocessRecord:
    """
    Utility that stores incoming (client state) and outgoing (keyframe) data such as it can be used by concurrent threads.
    """

    def __init__(self, networking_config, max_steps_ahead: int) -> None:
        self._networking_config = networking_config
        self._keyframe_queue: Queue = Queue()
        self._client_state_queue: Queue = Queue()
        self._step_semaphore: Semaphore = create_semaphore(max_steps_ahead)
        self._connection_record_queue: Queue = Queue()

    def send_keyframe_to_networking_thread(self, keyframe) -> None:
        """Send a keyframe (outgoing data) to the networking thread."""
        # Acquire the semaphore to ensure the simulation doesn't advance too far ahead
        self._step_semaphore.acquire()
        self._keyframe_queue.put(keyframe)

    def send_client_state_to_main_thread(self, client_state) -> None:
        """Send a client state (incoming data) to the main thread."""
        self._client_state_queue.put(client_state)

    def send_connection_record_to_main_thread(self, connection_record) -> None:
        """Send a connection record to the main thread."""
        assert "connectionId" in connection_record
        assert "isClientReady" in connection_record
        self._connection_record_queue.put(connection_record)

    def get_queued_keyframes(self) -> List[Any]:
        """Dequeue all keyframes."""
        keyframes = []

        while not self._keyframe_queue.empty():
            keyframe = self._keyframe_queue.get(block=False)
            keyframes.append(keyframe)
            self._step_semaphore.release()

        return keyframes

    def get_single_queued_keyframe(self) -> Optional[Any]:
        """Dequeue one keyframe."""
        if self._keyframe_queue.empty():
            return None

        keyframe = self._keyframe_queue.get(block=False)
        self._step_semaphore.release()
        return keyframe

    @staticmethod
    def _dequeue_all(queue: Queue) -> List[Any]:
        """Dequeue all items from a queue."""
        items = []

        while not queue.empty():
            item = queue.get(block=False)
            items.append(item)

        return items

    def get_queued_client_states(self) -> List[Any]:
        """Dequeue all client states."""
        return self._dequeue_all(self._client_state_queue)

    def get_queued_connection_records(self) -> List[Any]:
        """Dequeue all connection records."""
        return self._dequeue_all(self._connection_record_queue)
