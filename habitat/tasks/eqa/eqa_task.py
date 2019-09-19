#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional

import attr
import numpy as np
from gym import spaces, Space

from habitat.config import Config
from habitat.core.embodied_task import Action, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Observations, Sensor, SensorTypes, Simulator
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav_task import NavigationEpisode, NavigationTask


@attr.s(auto_attribs=True)
class QuestionData:
    question_text: str
    answer_text: str
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
    def __init__(self, dataset, **kwargs):
        self.uuid = "question"
        self.sensor_type = SensorTypes.TEXT
        self._dataset = dataset

        self.observation_space = spaces.Discrete(
            len(dataset.get_questions_vocabulary())
        )

    def get_observation(
        self,
        observations: Dict[str, Observations],
        episode: EQAEpisode,
        **kwargs,
    ):
        return self._dataset.get_questions_vocabulary()[
            episode.question.question_text
        ]


@registry.register_measure
class CorrectAnswer(Measure):
    """CorrectAnswer
    """

    def __init__(self, dataset, **kwargs):
        self._dataset = dataset
        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "correct_answer"

    def reset_metric(self, episode):
        self._metric = self._dataset.get_answers_vocabulary()[
            episode.question.answer_text
        ]

    def update_metric(self, *args: Any, **kwargs: Any):
        pass


@registry.register_measure
class EpisodeInfo(Measure):
    """Episode Info
    """

    def __init__(self, sim, config, **kwargs):
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "episode_info"

    def reset_metric(self, episode):
        self._metric = vars(episode).copy()

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        pass


@registry.register_measure
class ActionStats(Measure):
    """Actions Statistic
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "action_stats"

    def reset_metric(self, episode):
        self._metric = None

    def update_metric(self, episode, action):
        self._metric = {"previous_action": action}


@registry.register_measure
class AnswerAccuracy(Measure):
    """AnswerAccuracy
    """

    def __init__(self, dataset, task, **kwargs):
        self._dataset = dataset
        self._task = task
        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "answer_accuracy"

    def reset_metric(self, episode):
        self._metric = 0

    def update_metric(
        self, action=None, episode=None, *args: Any, **kwargs: Any
    ):
        if episode is None:
            return

        if action["action"] == AnswerAction.name:
            self._metric = (
                1
                if self._dataset.get_answers_vocabulary()[
                    episode.question.answer_text
                ]
                == action["action_args"]["answer_id"]
                else 0
            )

@registry.register_measure
class DistanceToGoal(Measure):
    """Distance To Goal metrics
    """

@registry.register_measure
class DistanceToGoal(Measure):
    """Distance To Goal metrics
    """

    def __init__(self, sim: Simulator, config: Config, **kwargs):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "distance_to_goal"

    def reset_metric(self, episode):
        self._previous_position = self._sim.get_agent_state().position.tolist()
        self._start_end_episode_distance = self._sim.geodesic_distance(
            self._previous_position, episode.goals[0].position
        )
        self._agent_episode_distance = 0.0
        self._metric = None

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(self, episode, action):
        current_position = self._sim.get_agent_state().position.tolist()

        distance_to_target = self._sim.geodesic_distance(
            current_position, episode.goals[0].position
        )

        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = {
            "distance_to_target": distance_to_target,
            "start_distance_to_target": self._start_end_episode_distance,
            "distance_delta": self._start_end_episode_distance
            - distance_to_target,
            "agent_path_length": self._agent_episode_distance,
        }

        self._metric = {
            "distance_to_target": distance_to_target,
            "start_distance_to_target": self._start_end_episode_distance,
            "distance_delta": self._start_end_episode_distance
            - distance_to_target,
            "agent_path_length": self._agent_episode_distance,
        }

@registry.register_task(name="EQA-v0")
class EQATask(NavigationTask):
    """
        Embodied Question Answering Task
        Usage example:
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
        self._task = task
        self._sim = sim
        self._task.answer = None
        self._task.is_stopped = False
        self._dataset = dataset

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.name

    def reset(self, *args: Any, **kwargs: Any) -> None:
        self._task.answer = None
        self._task.is_stopped = False
        self._task.is_valid = True
        return

    def step(
        self, *args: Any, answer_id: int, **kwargs: Any
    ) -> Dict[str, Observations]:
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        if self._task.answer is not None:
            self._task.is_valid = False
            self._task.invalid_reason = "Agent answered question twice."

        self._task.answer = answer_id
        return self._sim.get_observations_at()

    def get_action_space(self):
        r"""
        Returns:
             the current metric for ``Measure``.
        """
        return spaces.Dict(
            {
                "answer_id": spaces.Discrete(
                    n=len(self._dataset.get_answers_vocabulary())
                )
            }
        )
