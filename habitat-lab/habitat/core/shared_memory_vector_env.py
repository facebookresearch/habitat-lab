import signal
import numpy as np
from multiprocessing import shared_memory, Event
from multiprocessing.connection import Connection
from queue import Queue
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from habitat.core.vector_env import VectorEnv


class SharedMemoryPool:
    def __init__(self, num_blocks: int, block_size: int, dtype: str):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.dtype = dtype
        self.pool = []
        self.lock = Lock()

        for _ in range(num_blocks):
            shm = shared_memory.SharedMemory(create=True, size=block_size)
            self.pool.append(shm)

    def acquire(self) -> shared_memory.SharedMemory:
        with self.lock:
            if not self.pool:
                raise RuntimeError("Shared memory pool exhausted!")
            return self.pool.pop()

    def release(self, shm: shared_memory.SharedMemory) -> None:
        with self.lock:
            self.pool.append(shm)

    def close(self):
        for shm in self.pool:
            shm.close()
            shm.unlink()

class SharedMemoryVectorEnv(VectorEnv):
    def __init__(
        self,
        make_env_fn: Callable[..., Any],
        env_fn_args: Sequence[Tuple],
        num_shared_blocks: int = 10,
        shared_block_size: int = 10_000_000,  # Adjust as needed
        auto_reset_done: bool = True,
        multiprocessing_start_method: str = "forkserver",
        workers_ignore_signals: bool = False,
    ):
        super().__init__(
            make_env_fn,
            env_fn_args,
            auto_reset_done=auto_reset_done,
            multiprocessing_start_method=multiprocessing_start_method,
            workers_ignore_signals=workers_ignore_signals,
        )
        # Initialize shared memory pools
        self.shared_mem_pool = SharedMemoryPool(
            num_blocks=num_shared_blocks,
            block_size=shared_block_size,
            dtype="uint8",
        )

    def _serialize_data(self, data: Any) -> Any:
        """
        Serialize data, replacing large arrays and strings with shared memory placeholders.
        """
        if isinstance(data, dict):
            return {key: self._serialize_data(value) for key, value in data.items()}
        elif isinstance(data, np.ndarray):
            shm = self.shared_mem_pool.acquire()
            shm_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf[: data.nbytes])
            np.copyto(shm_array, data)
            return {
                "__shared_array__": True,
                "name": shm.name,
                "shape": data.shape,
                "dtype": str(data.dtype),
            }
        elif isinstance(data, str) and len(data) > 100:  # Handle large strings
            shm = self.shared_mem_pool.acquire()
            encoded = data.encode("utf-8")
            shm.buf[: len(encoded)] = encoded
            return {"__shared_string__": True, "name": shm.name, "size": len(encoded)}
        else:
            return data

    def _deserialize_data(self, data: Any) -> Any:
        """
        Deserialize data, replacing shared memory placeholders with the original data.
        """
        if isinstance(data, dict):
            if "__shared_array__" in data:
                shm = shared_memory.SharedMemory(name=data["name"])
                array = np.ndarray(
                    data["shape"], dtype=data["dtype"], buffer=shm.buf[: np.prod(data["shape"])]
                )
                result = array.copy()
                self.shared_mem_pool.release(shm)  # Return to pool
                return result
            elif "__shared_string__" in data:
                shm = shared_memory.SharedMemory(name=data["name"])
                encoded = bytes(shm.buf[: data["size"]])
                result = encoded.decode("utf-8")
                self.shared_mem_pool.release(shm)  # Return to pool
                return result
            else:
                return {key: self._deserialize_data(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._deserialize_data(item) for item in data]
        else:
            return data

    def step(self, data: Sequence[Any]) -> List[Any]:
        """
        Override step to use shared memory serialization.
        """
        serialized_actions = [self._serialize_data(action) for action in data]
        for write_fn, action in zip(self._connection_write_fns, serialized_actions):
            write_fn((STEP_COMMAND, action))

        results = []
        for read_fn in self._connection_read_fns:
            obs, reward, done, info = read_fn()
            obs = self._deserialize_data(obs)
            results.append((obs, reward, done, info))
        return results

    def reset(self) -> List[Any]:
        """
        Override reset to use shared memory serialization.
        """
        for write_fn in self._connection_write_fns:
            write_fn((RESET_COMMAND, None))
        results = []
        for read_fn in self._connection_read_fns:
            obs = read_fn()
            obs = self._deserialize_data(obs)
            results.append(obs)
        return results

    def close(self) -> None:
        """
        Close environments and clean up shared memory.
        """
        super().close()
        self.shared_mem_pool.close()
