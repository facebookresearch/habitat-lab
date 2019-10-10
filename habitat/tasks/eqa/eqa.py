#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional

import attr
import numpy as np
from gym import Space, spaces

from habitat.config import Config
from habitat.core.embodied_task import Action, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Observations, Sensor, SensorTypes, Simulator
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import NavigationEpisode, NavigationTask


@attr.s(auto_attribs=True)
class QuestionData:
    question_text: str
    answer_text: str
    question_tokens: Optional[List[str]] = None
    answer_tokens: Optional[List[str]] = None
    question_type: Optional[str] = None


@attr.s(auto_attribs=True, kw_only=True)
class EQAEpisode(NavigationEpisode):
    r"""Specification of episode that includes initial position and rotation of
    agent, goal, question specifications and optional shortest paths.

    Args:
        scene_id: id of scene inside the simulator.
        start_position: numpy ndarray containing 3 entries for (x, y, z).
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation.
        goals: relevant goal object/room.
        question: question related to goal object.
    """

    question: QuestionData = attr.ib(
        default=None, validator=not_none_validator
    )


@registry.register_sensor
class QuestionSensor(Sensor):
    def __init__(self, dataset, *args: Any, **kwargs: Any):
        self.uuid = "question"
        self.sensor_type = SensorTypes.TEXT
        self._dataset = dataset
        self.observation_space = spaces.Discrete(
            dataset.question_vocab.get_size()
        )

    def get_observation(
        self,
        observations: Dict[str, Observations],
        episode: EQAEpisode,
        *args: Any,
        **kwargs: Any
    ):
        return episode.question.question_tokens


@registry.register_measure
class CorrectAnswer(Measure):
    """CorrectAnswer
    """

    def __init__(self, dataset, *args: Any, **kwargs: Any):
        self._dataset = dataset
        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "correct_answer"

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._metric = episode.question.answer_tokens

    def update_metric(self, *args: Any, **kwargs: Any):
        pass


@registry.register_measure
class EpisodeInfo(Measure):
    """Episode Info
    """

    def __init__(self, sim, config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "episode_info"

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._metric = vars(episode).copy()

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        pass


@registry.register_measure
class AnswerAccuracy(Measure):
    """AnswerAccuracy
    """

    def __init__(self, dataset, *args: Any, **kwargs: Any):
        self._dataset = dataset
        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "answer_accuracy"

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._metric = 0

    def update_metric(
        self, action=None, episode=None, *args: Any, **kwargs: Any
    ):
        if episode is None:
            return

        if action["action"] == AnswerAction.name:
            self._metric = (
                1
                if episode.question.answer_tokens
                == action["action_args"]["answer_id"]
                else 0
            )


@registry.register_task(name="EQA-v0")
class EQATask(NavigationTask):
    """
        Embodied Question Answering Task
        Usage example:
            env = habitat.Env(config=eqa_config)

            env.reset()

            for i in range(10):
                action = sample_non_stop_action(env.action_space)
                if action["action"] != AnswerAction.name:
                    env.step(action)
                metrics = env.get_metrics() # to check distance to target

            correct_answer_id = env.current_episode.question.answer_tokens
            env.step(
                {
                    "action": AnswerAction.name,
                    "action_args": {"answer_id": correct_answer_id},
                }
            )

            metrics = env.get_metrics()
            assert metrics["answer_accuracy"] == 1


            observations = self._env.reset()
            while not env.episode_over:
                action = agent.act(observations)
                observations = env.step(action)
            env.task.answer_question(
                agent.answer_question(observations), env.episode_over)
            metrics = self._env.get_metrics()
    """

    def _check_episode_is_active(
        self, *args, action, episode, action_args=None, **kwargs
    ) -> bool:
        return self.is_valid and self.answer is None


@registry.register_task_action
class AnswerAction(Action):
    _answer: Optional[str]
    name: str = "ANSWER"

    def __init__(self, *args: Any, sim, task, dataset, **kwargs: Any) -> None:
        self._sim = sim
        self._dataset = dataset

    def reset(self, task: EQATask, *args: Any, **kwargs: Any) -> None:
        task.answer = None
        task.is_valid = True
        return

    def step(
        self, *args: Any, answer_id: int, task: EQATask, **kwargs: Any
    ) -> Dict[str, Observations]:
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        if task.answer is not None:
            task.is_valid = False
            task.invalid_reason = "Agent answered question twice."

        task.answer = answer_id
        return self._sim.get_observations_at()

    @property
    def action_space(self) -> Space:
        r"""
        Returns:
             the current metric for ``Measure``.
        """
        return spaces.Dict(
            {
                "answer_id": spaces.Discrete(
                    self._dataset.answer_vocab.get_size()
                )
            }
        )
