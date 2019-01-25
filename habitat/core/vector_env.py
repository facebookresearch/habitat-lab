import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import List, Tuple, Callable, Union, Any, Set

import habitat
import numpy as np
from habitat.core.env import Env, Observations
from habitat.core.logging import logger
from habitat.core.utils import tile_images
from yacs.config import CfgNode

STEP_COMMAND = 'step'
RESET_COMMAND = 'reset'
RENDER_COMMAND = 'render'
CLOSE_COMMAND = 'close'
OBSERVATION_SPACE_COMMAND = 'observation_space'
ACTION_SPACE_COMMAND = 'action_space'


def _worker_env(worker_connection: Connection, env_fn: Callable,
                env_fn_args: Tuple[Any], auto_reset_done: bool) -> None:
    r"""Process worker for creating and interacting with the environment.
    """
    env = env_fn(*env_fn_args)
    try:
        command, data = worker_connection.recv()
        while command != CLOSE_COMMAND:
            if command == STEP_COMMAND:

                # different step methods for habitat.RLEnv and habitat.Env
                if isinstance(env, habitat.RLEnv):
                    # habitat.RLEnv
                    observations, reward, done, info = env.step(data)
                    if auto_reset_done and done:
                        observations = env.reset()
                    worker_connection.send((observations, reward, done, info))
                elif isinstance(env, habitat.Env):
                    # habitat.Env
                    observations = env.step(data)
                    if auto_reset_done and env.episode_over:
                        observations = env.reset()
                    worker_connection.send(observations)
                else:
                    raise NotImplementedError

            elif command == RESET_COMMAND:

                observations = env.reset()
                worker_connection.send(observations)

            elif command == RENDER_COMMAND:

                worker_connection.send(env.render(*data[0], **data[1]))

            elif command == OBSERVATION_SPACE_COMMAND or \
                    command == ACTION_SPACE_COMMAND:

                worker_connection.send(getattr(env, command))

            else:
                raise NotImplementedError

            command, data = worker_connection.recv()

        worker_connection.close()
    except KeyboardInterrupt:
        logger.info('Worker KeyboardInterrupt')
    finally:
        env.close()


def _make_env_fn(config: CfgNode, dataset: habitat.Dataset,
                 rank: int = 0) -> Env:
    r"""Constructor for default habitat Env.
    :param config: configurations for environment
    :param dataset: dataset for environment
    :param rank: rank for setting seeds for environment
    :return: constructed habitat Env
    """
    habitat_env = Env(config=config, dataset=dataset)
    habitat_env.seed(config.seed + rank)
    return habitat_env


class VectorEnv:
    def __init__(self,
                 make_env_fn: Callable[..., Env] = _make_env_fn,
                 env_fn_args: Tuple[Tuple] = None,
                 auto_reset_done: bool = True,
                 multiprocessing_start_method: str = 'forkserver') -> None:
        r"""
        :param make_env_fn: Function which creates a single environment.
        :param env_fn_args: tuple of tuple of args to pass to the make_env_fn.
        :param auto_reset_done: automatically reset the environment when
                                done. This functionality is provided for
                                seamless training of vectorized environments.
        :param multiprocessing_start_method: The multiprocessing method used to
                                             spawn worker processes
                                             Valid methods are
                                             ``{'spawn', 'forkserver',
                                                'fork'}``
                                             ``'forkserver'`` is the
                                             recommended method as it works
                                             well with CUDA.
                                             If ``'fork'`` is used,
                                             the subproccess must be started
                                             before any GPU useage
        """
        self._is_waiting = False

        assert env_fn_args is not None and len(env_fn_args) > 0, \
            "number of environments to be created should be greater than 0"

        self._num_envs = len(env_fn_args)

        assert multiprocessing_start_method in self._valid_start_methods, \
            ("multiprocessing_start_method must be one of {}. Got '{}'"
             ).format(self._valid_start_methods, multiprocessing_start_method)
        self._auto_reset_done = auto_reset_done
        mp_ctx = mp.get_context(multiprocessing_start_method)

        self._parent_connections, self._worker_connections = \
            zip(*[mp_ctx.Pipe(duplex=True) for _ in range(self._num_envs)])
        self._processes: List[mp.Process] = []
        for worker_conn, parent_conn, env_args in zip(
                self._worker_connections, self._parent_connections,
                env_fn_args):
            ps = mp_ctx.Process(
                target=_worker_env,
                args=(worker_conn, make_env_fn, env_args,
                      self._auto_reset_done))
            self._processes.append(ps)
            ps.daemon = True
            ps.start()
            worker_conn.close()

        for parent_conn in self._parent_connections:
            parent_conn.send((OBSERVATION_SPACE_COMMAND, None))
        self.observation_spaces = [parent_conn.recv() for parent_conn
                                   in self._parent_connections]
        for parent_conn in self._parent_connections:
            parent_conn.send((ACTION_SPACE_COMMAND, None))
        self.action_spaces = [parent_conn.recv() for parent_conn
                              in self._parent_connections]

    def reset(self):
        r"""Reset all the _num_envs in the vector
        :return: [observations] * (_num_envs)
        """
        self._is_waiting = True
        for parent_conn in self._parent_connections:
            parent_conn.send((RESET_COMMAND, None))
        results = []
        for parent_conn in self._parent_connections:
            results.append(parent_conn.recv())
        self._is_waiting = False
        return results

    def reset_at(self, index_env: int):
        r"""Reset only the index_env environmet in the vector
        :param index_env: index of the environment to be reset
        :return: [observations]
        """
        self._is_waiting = True
        self._parent_connections[index_env].send((RESET_COMMAND, None))
        results = [self._parent_connections[index_env].recv()]
        self._is_waiting = False
        return results

    def step_at(self, index_env: int, action: int):
        r"""
        :param index_env: index of the environment to be stepped into
        :param action: action to be taken
        :return: [observations] for the indexed environment
        """
        self._is_waiting = True
        self._parent_connections[index_env].send((STEP_COMMAND, action))
        results = [self._parent_connections[index_env].recv()]
        self._is_waiting = False
        return results

    def async_step(self, actions: List[int]) -> None:
        r"""Asynchronously step in the environments.
        """
        self._is_waiting = True
        for parent_conn, action in zip(self._parent_connections, actions):
            parent_conn.send((STEP_COMMAND, action))

    def wait_step(self) -> List[Observations]:
        r"""Wait until all the asynchronized environments have synchronized.
        """
        observations = []
        for parent_conn in self._parent_connections:
            observations.append(parent_conn.recv())
        self._is_waiting = False
        return observations

    def step(self, actions: List[int]):
        r"""
        :param actions: list of size _num_envs containing action to be taken
                        in each environment.
        :return: [observations] * _num_envs
        """
        self.async_step(actions)
        return self.wait_step()

    def close(self) -> None:
        if self._is_waiting:
            for parent_conn in self._parent_connections:
                parent_conn.recv()
        for parent_conn in self._parent_connections:
            parent_conn.send((CLOSE_COMMAND, None))
        for process in self._processes:
            process.join()

    def render(self, mode: str = 'human', *args, **kwargs) \
            -> Union[np.ndarray, None]:
        r"""Render observations from all environments in a tiled image.
        """
        for parent_conn in self._parent_connections:
            parent_conn.send((RENDER_COMMAND, (args, {
                'mode': 'rgb_array',
                **kwargs
            })))
        images = [pipe.recv() for pipe in self._parent_connections]
        tile = tile_images(images)
        if mode == 'human':
            import cv2
            cv2.imshow('vecenv', tile[:, :, ::-1])
            cv2.waitKey(1)
            return None
        elif mode == 'rgb_array':
            return tile
        else:
            raise NotImplementedError

    @property
    def _valid_start_methods(self) -> Set[str]:
        return {'forkserver', 'spawn', 'fork'}
