import dataclasses
import os
import queue

import a0


@dataclasses.dataclass(order=True)
class ReplayMessage:
    timestamp: int
    pkt: a0.Packet = dataclasses.field(compare=False)
    path: str = dataclasses.field(compare=False)
    topic: str = dataclasses.field(compare=False)
    reader: a0.ReaderSync = dataclasses.field(compare=False)


class ReplayManager:
    def __init__(self, read_paths):
        self._srcs = {
            read_path: a0.ReaderSync(a0.File(read_path), a0.INIT_OLDEST)
            for read_path in read_paths
        }

        self._pq = queue.PriorityQueue()
        for path, reader in self._srcs.items():
            self._load_next_message(path, reader)

    def _load_next_message(self, path, reader):
        if not reader.can_read():
            return

        pkt = reader.read()
        timestamp = int(dict(pkt.headers)["a0_time_mono"])
        topic = os.path.basename(path).split(".")[0]
        self._pq.put(ReplayMessage(timestamp, pkt, path, topic, reader))

    def can_read(self):
        return not self._pq.empty()

    def read(self):
        replay_message = self._pq.get()
        self._load_next_message(replay_message.path, replay_message.reader)
        return replay_message

    def __iter__(self):
        return self

    def __next__(self):
        if not self.can_read():
            raise StopIteration
        return self.read()
