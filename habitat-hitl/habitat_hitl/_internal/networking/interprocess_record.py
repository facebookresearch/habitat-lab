#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from multiprocessing import Queue
from typing import List, Optional

from habitat_hitl.core.types import (
    ClientState,
    ConnectionRecord,
    DataDict,
    DisconnectionRecord,
    Keyframe,
)


class InterprocessRecord:
    """
    Utility that stores incoming (client state) and outgoing (keyframe) data such as it can be used by concurrent threads.
    """

    def __init__(self, networking_config) -> None:
        self._networking_config = networking_config
        self._keyframe_queue: Queue[Keyframe] = Queue()
        self._client_state_queue: Queue[ClientState] = Queue()
        self._connection_record_queue: Queue[ConnectionRecord] = Queue()
        self._disconnection_record_queue: Queue[DisconnectionRecord] = Queue()

    def send_keyframe_to_networking_thread(self, keyframe: Keyframe) -> None:
        """Send a keyframe (outgoing data) to the networking thread."""
        # Acquire the semaphore to ensure the simulation doesn't advance too far ahead
        self._keyframe_queue.put(keyframe)

    def send_client_state_to_main_thread(
        self, client_state: ClientState
    ) -> None:
        """Send a client state (incoming data) to the main thread."""
        self._client_state_queue.put(client_state)

    def send_connection_record_to_main_thread(
        self, connection_record: ConnectionRecord
    ) -> None:
        """Send a connection record to the main thread."""
        assert "connectionId" in connection_record
        assert "isClientReady" in connection_record
        self._connection_record_queue.put(connection_record)

    def send_disconnection_record_to_main_thread(
        self, disconnection_record: DisconnectionRecord
    ) -> None:
        """Send a disconnection record to the main thread."""
        assert "connectionId" in disconnection_record
        self._disconnection_record_queue.put(disconnection_record)

    def get_single_queued_keyframe(self) -> Optional[Keyframe]:
        """Dequeue one keyframe."""
        if self._keyframe_queue.empty():
            return None

        keyframe = self._keyframe_queue.get(block=False)
        return keyframe

    @staticmethod
    def _dequeue_all(queue: Queue) -> List[DataDict]:
        """Dequeue all items from a queue."""
        items = []

        while not queue.empty():
            item = queue.get(block=False)
            items.append(item)

        return items

    def get_queued_keyframes(self) -> List[Keyframe]:
        """Dequeue all keyframes."""
        return self._dequeue_all(self._keyframe_queue)

    def get_queued_client_states(self) -> List[ClientState]:
        """Dequeue all client states."""
        return self._dequeue_all(self._client_state_queue)

    def get_queued_connection_records(self) -> List[ConnectionRecord]:
        """Dequeue all connection records."""
        return self._dequeue_all(self._connection_record_queue)

    def get_queued_disconnection_records(self) -> List[DisconnectionRecord]:
        """Dequeue all disconnection records."""
        return self._dequeue_all(self._disconnection_record_queue)
