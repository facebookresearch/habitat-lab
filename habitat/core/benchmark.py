#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""Implements evaluation of ``habitat.Agent`` inside ``habitat.Env``.
``habitat.Benchmark`` creates a ``habitat.Env`` which is specified through
the ``config_env`` parameter in constructor. The evaluation is task agnostic
and is implemented through metrics defined for ``habitat.EmbodiedTask``.
"""

import os
from collections import defaultdict
from typing import Dict, Optional

from habitat.config.default import get_config
from habitat.core.agent import Agent
from habitat.core.env import Env

import numpy as np
import time
import sys


class Benchmark:
    r"""Benchmark for evaluating agents in environments."""

    def __init__(
        self, config_paths: Optional[str] = None, eval_remote: bool = False
    ) -> None:
        r"""..

        :param config_paths: file to be used for creating the environment
        :param eval_remote: boolean indicating whether evaluation should be run remotely or locally
        """
        config_env = get_config(config_paths)
        self._eval_remote = eval_remote

        if self._eval_remote is True:
            self._env = None
        else:
            self._env = Env(config=config_env)

    def remote_evaluate(
        self, agent: "Agent", num_episodes: Optional[int] = None
    ):
        # The modules imported below are specific to habitat-challenge remote evaluation.
        # These modules are not part of the habitat-lab repository.
        import pickle
        import time

        import evalai_environment_habitat  # noqa: F401
        import evaluation_pb2
        import evaluation_pb2_grpc
        import grpc

        time.sleep(60)

        def pack_for_grpc(entity):
            return pickle.dumps(entity)

        def unpack_for_grpc(entity):
            return pickle.loads(entity)

        def remote_ep_over(stub):
            res_env = unpack_for_grpc(
                stub.episode_over(evaluation_pb2.Package()).SerializedEntity
            )
            return res_env["episode_over"]

        env_address_port = os.environ.get("EVALENV_ADDPORT", "localhost:8085")
        channel = grpc.insecure_channel(env_address_port)
        stub = evaluation_pb2_grpc.EnvironmentStub(channel)

        base_num_episodes = unpack_for_grpc(
            stub.num_episodes(evaluation_pb2.Package()).SerializedEntity
        )
        num_episodes = base_num_episodes["num_episodes"]

        agg_metrics: Dict = defaultdict(float)

        count_episodes = 0

        while count_episodes < num_episodes:
            agent.reset()
            res_env = unpack_for_grpc(
                stub.reset(evaluation_pb2.Package()).SerializedEntity
            )

            while not remote_ep_over(stub):
                obs = res_env["observations"]
                action = agent.act(obs)

                res_env = unpack_for_grpc(
                    stub.act_on_environment(
                        evaluation_pb2.Package(
                            SerializedEntity=pack_for_grpc(action)
                        )
                    ).SerializedEntity
                )

            metrics = unpack_for_grpc(
                stub.get_metrics(
                    evaluation_pb2.Package(
                        SerializedEntity=pack_for_grpc(action)
                    )
                ).SerializedEntity
            )

            for m, v in metrics["metrics"].items():
                agg_metrics[m] += v
            count_episodes += 1

        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

        stub.evalai_update_submission(evaluation_pb2.Package())

        return avg_metrics

    def local_evaluate(
        self, agent: "Agent", num_episodes: Optional[int] = None,
        skip_first_n: Optional[int] = 0
    ) -> Dict[str, float]:
        if num_episodes is None:
            num_episodes = len(self._env.episodes)
        else:
            assert num_episodes <= len(self._env.episodes), (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(
                    num_episodes, len(self._env.episodes)
                )
            )

        assert num_episodes > 0, "num_episodes should be greater than 0"

        agg_metrics: Dict = defaultdict(float)
        episode_metrics = dict(scene=[], num_steps=[], time=[], num_collisions=[], xy_error=[], spl=[], softspl=[], success=[])
        last_success = None

        for _ in range(skip_first_n):
            self._env.reset()

        count_episodes = 0
        try:
            while count_episodes < num_episodes:
                agent.reset(last_success=last_success)
                observations = self._env.reset()

                # Skip infeasible episodes. This is easy to detect using spl
                # because start-to-path is recomputed and if it is infinite
                # spl will become nan.
                while not np.isfinite(self._env.get_metrics()['spl']):
                    observations = self._env.reset()

                # metrics = self._env.get_metrics()
                action = None
                num_steps = 0
                t = time.time()

                while not self._env.episode_over:
                    action = agent.act(observations)
                    observations = self._env.step(action)
                    num_steps += 1
                    # metrics = self._env.get_metrics()
                episode_time = time.time() - t

                metrics = self._env.get_metrics()
                if isinstance(action, dict) and 'xy_error' in action.keys():
                    # Add all outputs to metrics
                    for key, val in action.items():
                        try:
                            metrics[str(key)] = float(val)
                        except TypeError:
                            pass
                    xy_error = action['xy_error']
                else:
                    xy_error = 999.
                metrics['small_error'] = 1. if xy_error < 7.2 else 0.  # 0.36 / 0.05
                metrics['episode_length'] = num_steps
                metrics['time'] = episode_time
                if 'softspl' not in metrics.keys():
                    metrics['softspl'] = 0.

                for m, v in metrics.items():
                    if m != 'top_down_map':
                        agg_metrics[m] += v
                count_episodes += 1

                episode_metrics['scene'].append(self._env.current_episode.scene_id)
                episode_metrics['num_steps'].append(num_steps)
                episode_metrics['time'].append(episode_time)
                episode_metrics['xy_error'].append(xy_error)
                episode_metrics['spl'].append(metrics['spl'])
                episode_metrics['softspl'].append(metrics['softspl'])
                episode_metrics['success'].append(metrics['success'])
                last_success = metrics['success']

                print ("%d/%d: Meann success: %f, spl: %f. err: %f This trial success: %f. SPL: %f  SOFT_SPL: %f. err %f"%(
                    count_episodes, num_episodes,
                    agg_metrics['success'] / count_episodes, agg_metrics['spl'] / count_episodes,
                    agg_metrics['xy_error'] / count_episodes,
                    metrics['success'], metrics['spl'], metrics['softspl'],
                    metrics['xy_error']))

            # One more agent reset, so last data can be saved.
            agent.reset()

        except KeyboardInterrupt:
            print ("interrupt")

        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}
        avg_metrics['num_episodes'] = count_episodes
        avg_metrics['input_args'] = ' '.join([str(x) for x in sys.argv])

        try:
            print (avg_metrics['input_args'])
            print (agent.params.name)
        except:
            pass


        import json
        timestamp_str = time.strftime('%m-%d-%H-%M-%S', time.localtime())
        filename = './temp/evals/eval_{}'.format(timestamp_str)
        with open(filename + '.summary.json', 'w') as file:
            json.dump(avg_metrics, file, indent=4)
        print (filename + '.summary.json')
        with open(filename + '.episodes.json', 'w') as file:
            json.dump(episode_metrics, file, indent=4)
        print (filename + '.episodes.json')

        return avg_metrics

    def evaluate(
        self, agent: "Agent", num_episodes: Optional[int] = None,
        skip_first_n: Optional[int] = 0
    ) -> Dict[str, float]:
        r"""..

        :param agent: agent to be evaluated in environment.
        :param num_episodes: count of number of episodes for which the
            evaluation should be run.
        :return: dict containing metrics tracked by environment.
        """

        if self._eval_remote is True:
            return self.remote_evaluate(agent, num_episodes)
        else:
            return self.local_evaluate(agent, num_episodes, skip_first_n=skip_first_n)
