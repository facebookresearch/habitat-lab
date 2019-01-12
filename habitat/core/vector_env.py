import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import List, Tuple, Callable, Union, Any

import habitat
import numpy as np
from habitat.core.env import Env, Observation
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
        while True:
            command, data = worker_connection.recv()
            if command == STEP_COMMAND:
                observation, reward, done, info = env.step(data)
                if auto_reset_done and done:
                    observation = env.reset()
                worker_connection.send((observation, reward, done, info))
            elif command == RESET_COMMAND:
                observation = env.reset()
                worker_connection.send(observation)
            elif command == RENDER_COMMAND:
                worker_connection.send(env.render(*data[0], **data[1]))
            elif command == CLOSE_COMMAND:
                worker_connection.close()
                break
            elif command == OBSERVATION_SPACE_COMMAND or \
                    command == ACTION_SPACE_COMMAND:
                worker_connection.send(getattr(env, command))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('VectorEnv worker KeyboardInterrupt')
    finally:
        env.close()


class VectorEnv:
    def __init__(self, configs: List[CfgNode], datasets: List[habitat.Dataset],
                 auto_reset_done: bool = True) -> None:
        r"""
        :param configs: list containing configurations for environments.
        :param datasets: list of datasets for environments
        :param auto_reset_done: automatically reset the environment when
                                done. This functionality is provided for
                                seamless training of vectorized environments.
        """
        self._is_waiting = False
        assert len(configs) > 0, "number of environments to be created " \
                                 "should be greater than 0"
        assert len(configs) == len(datasets), "mismatch between number of " \
                                              "configs and datasets"
        self._num_envs = len(configs)
        self._auto_reset_done = auto_reset_done
        mp_ctx = mp.get_context('forkserver')

        make_env_fn_args = [
            (configs[i], datasets[i], i) for i in range(self._num_envs)
        ]

        self._parent_connections, self._worker_connections = \
            zip(*[mp_ctx.Pipe(duplex=True) for _ in range(self._num_envs)])
        self._processes: List[mp.Process] = []
        for worker_conn, parent_conn, env_fn_args in zip(
                self._worker_connections, self._parent_connections,
                make_env_fn_args):
            ps = mp_ctx.Process(
                target=_worker_env,
                args=(worker_conn, self._make_env_fn, env_fn_args,
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

    @staticmethod
    def _make_env_fn(config: CfgNode, dataset: habitat.Dataset,
                     rank: int) -> Env:
        habitat_env = Env(config=config, dataset=dataset)
        habitat_env.seed(config.seed + rank)
        return habitat_env

    def reset(self) -> List[Tuple[Observation, Observation, bool, None]]:
        r"""Reset all the _num_envs in the vector
        :return: [observation, reward, done, info] * (_num_envs)
        """
        self._is_waiting = True
        for parent_conn in self._parent_connections:
            parent_conn.send((RESET_COMMAND, None))
        results = []
        for parent_conn in self._parent_connections:
            results.append(parent_conn.recv())
        self._is_waiting = False
        return results

    def reset_at(self, index_env: int) -> \
            List[Tuple[Observation, Observation, bool, None]]:
        r"""Reset only the index_env environmet in the vector
        :param index_env: index of the environment to be reset
        :return: [observation, reward, done, info]
        """
        self._is_waiting = True
        self._parent_connections[index_env].send((RESET_COMMAND, None))
        results = [self._parent_connections[index_env].recv()]
        self._is_waiting = False
        return results

    def step_at(self, index_env: int, action: int) -> \
            List[Tuple[Observation, Observation, bool, None]]:
        r"""
        :param index_env: index of the environment to be stepped into
        :param action: action to be taken
        :return: [observation, reward, done, info] for the indexed environment
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

    def wait_step(self) -> Tuple[List[Observation], List[Observation],
                                 List[bool], List[None]]:
        r"""Wait until all the asynchronized environments have synchronized.
        """
        results = []
        for parent_conn in self._parent_connections:
            results.append(parent_conn.recv())
        self._is_waiting = False
        observations, rewards, dones, infos = zip(*results)
        return list(observations), list(rewards), list(dones), list(infos)

    def step(self, actions: List[int]) -> Tuple[List[Observation],
                                                List[Observation],
                                                List[bool], List[None]]:
        r"""
        :param actions: list of size _num_envs containing action to be taken
                        in each environment.
        :return: [observation, reward, done, info] * _num_envs
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
