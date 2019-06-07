#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import attr
import numpy as np
from gym import spaces

from habitat.core.registry import registry
from habitat.core.simulator import (
    Observations,
    Sensor,
    SensorSuite,
    SensorTypes,
)
from habitat.core.utils import not_none_validator
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
        self.dataset = dataset
        # TODO (maksymets) extend gym observation space for text and metadata
        self.observation_space = spaces.Discrete(
            len(dataset.get_questions_vocabulary()))

    def _get_observation(
        self,
        observations: Dict[str, Observations],
        episode: EQAEpisode,
        **kwargs
    ):
        return self.dataset.get_questions_vocabulary()[
            episode.question.question_text]

    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)

@registry.register_sensor
class AnswerSensor(Sensor):
    def __init__(self, dataset, **kwargs):
        self.uuid = "answer"
        self.sensor_type = SensorTypes.TEXT
        self.dataset = dataset
        # TODO (maksymets) extend gym observation space for text and metadata
        self.observation_space = spaces.Discrete(len(
            dataset.get_answers_vocabulary()))

    def _get_observation(
        self,
        observations: Dict[str, Observations],
        episode: EQAEpisode,
        **kwargs
    ):
        return self.dataset.get_answers_vocabulary()[
            episode.question.answer_text]

    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)

@registry.register_task(name="EQA-v0")
class EQATask(NavigationTask):
    pass
