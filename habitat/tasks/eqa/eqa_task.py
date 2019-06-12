#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Any

import attr
from gym import spaces

from habitat.core.registry import registry
from habitat.core.simulator import (
    Observations,
    Sensor,
    SensorTypes,
)
from habitat.core.utils import not_none_validator
from habitat.core.embodied_task import Measure
from habitat.tasks.nav.nav_task import NavigationEpisode, NavigationTask


# from habitat.datasets.eqa.mp3d_eqa_dataset import Matterport3dDatasetV1


@attr.s(auto_attribs=True)
class QuestionData:
    question_text: str
    answer_text: str
    question_type: Optional[str] = None


@attr.s(auto_attribs=True, kw_only=True)
class EQAEpisode(NavigationEpisode):
    """Specification of episode that includes initial position and rotation of
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
            len(dataset.get_questions_vocabulary()))

    def get_observation(
            self,
            observations: Dict[str, Observations],
            episode: EQAEpisode,
            **kwargs
    ):
        return self._dataset.get_questions_vocabulary()[
            episode.question.question_text]


@registry.register_sensor
class AnswerSensor(Sensor):
    def __init__(self, dataset, **kwargs):
        self.uuid = "answer"
        self.sensor_type = SensorTypes.TEXT
        self._dataset = dataset
        self.observation_space = spaces.Discrete(len(
            dataset.get_answers_vocabulary()))

    def get_observation(
            self,
            observations: Dict[str, Observations],
            episode: EQAEpisode,
            **kwargs
    ):
        return self._dataset.get_answers_vocabulary()[
            episode.question.answer_text]


@registry.register_measure
class EpisodeInfo(Measure):
    """Episode Info
    """

    def __init__(self, sim, config, dataset, **kwargs):
        self._sim = sim
        self._config = config
        self._dataset = dataset

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "episode_info"

    def reset_metric(self, episode):
        self._metric = episode

    def update_metric(self, episode, action):
        pass


@registry.register_measure
class AnswerAccuracy(Measure):
    """AnswerAccuracy
    """

    def __init__(self, sim, config, dataset, **kwargs):
        self._sim = sim
        self._config = config
        self._dataset = dataset

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "answer_accuracy"

    def reset_metric(self, episode):
        self._previous_position = self._sim.get_agent_state().position.tolist()
        self._start_end_episode_distance = episode.info["geodesic_distance"]
        self._agent_episode_distance = 0.0
        self._metric = 0

    def update_metric(self, episode, action):
        self._metric = self._dataset.get_answers_vocabulary()[
            episode.question.answer_text] == action


@registry.register_task(name="EQA-v0")
class EQATask(NavigationTask):
    def step(self, action):
        pass

    def answer_question(self, answer_id: int):
        episode = self.measurements.measures["episode_info"].get_metric()
        return self._dataset.get_answers_vocabulary()[
            episode.question.answer_text] == answer_id
