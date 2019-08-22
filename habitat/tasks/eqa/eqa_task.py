#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Type

import attr
import numpy as np
from gym import spaces, Space

from habitat.config import Config
from habitat.core.embodied_task import Measure
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
        self._metric = episode

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
        self._answer_received = False
        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "answer_accuracy"

    def reset_metric(self, episode):
        self._metric = 0
        self._answer_received = False

    def update_metric(
        self, action=None, episode=None, *args: Any, **kwargs: Any
    ):
        if (
            action.sim_action is None
            or action.task_action is None
            or episode is None
        ):
            return
        assert not self._answer_received, (
            "Question can be answered only " "once per episode."
        )
        self._answer_received = True
        if action.sim_action == SimulatorActions.STOP.value:
            self._metric = (
                1
                if self._dataset.get_answers_vocabulary()[
                    episode.question.answer_text
                ]
                == action.task_action
                else 0
            )
        else:
            # Episode wasn't finished, but to reflect unfinished episodes in
            # average accuracy metric we report 0
            self._metric = 0


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

    @registry.register_task_action(name="move_forward")
    def move_forward(self):
        self.sim.step(0)

    def step(self, action: int, episode):
        if action.sim_action is None:
            observations = super().step(action=action, episode=episode)
        else:
            observations = {}

        if action.task_action is None:
            observations.update(
                self.sensor_suite.get_observations(
                    observations=observations, episode=episode
                )
            )
        return observations

        # required_measures = {"episode_info", "action_stats", "answer_accuracy"}
        # assert (
        #     required_measures <= self.measurements.measures.keys()
        # ), f"{required_measures} are required to be enabled for EQA task"
        # episode = self.measurements.measures["episode_info"].get_metric()
        # sim_action = self.measurements.measures["action_stats"].get_metric()[
        #     "previous_action"
        # ]
        #
        # self.measurements.measures["answer_accuracy"].update_metric(
        #     answer_id=action.task_action,
        #     sim_action=action.sim_action,
        #     episode=episode,
        # )
        #
        # return self.sensor_suite.get_observations(
        #     observations=sim_observations, episode=episode
        # )

    @property
    def action_space(self) -> Space:
        return spaces.Dict(
            {
                "sim_action": self._sim.action_space,
                "task_action": self._sim.action_space,
            }
        )
