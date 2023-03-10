#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
from math import pi
from typing import Dict, Union

import numpy as np
from numpy import bool_, int64, ndarray

import habitat
from habitat.config.default import get_config
from habitat.core.simulator import Observations
from habitat.sims.habitat_simulator.actions import HabitatSimActions


class RandomAgent(habitat.Agent):
    def __init__(self, success_distance: float, goal_sensor_uuid: str) -> None:
        self.dist_threshold_to_stop = success_distance
        self.goal_sensor_uuid = goal_sensor_uuid

    def reset(self) -> None:
        pass

    def is_goal_reached(self, observations: Observations) -> bool_:
        dist = observations[self.goal_sensor_uuid][0]
        return dist <= self.dist_threshold_to_stop

    def act(self, observations: Observations) -> Dict[str, int64]:
        if self.is_goal_reached(observations):
            action = HabitatSimActions.stop
        else:
            action = np.random.choice(
                [
                    HabitatSimActions.move_forward,
                    HabitatSimActions.turn_left,
                    HabitatSimActions.turn_right,
                ]
            )
        return {"action": action}


class ForwardOnlyAgent(RandomAgent):
    def act(self, observations: Observations) -> Dict[str, int]:
        if self.is_goal_reached(observations):
            action = HabitatSimActions.stop
        else:
            action = HabitatSimActions.move_forward
        return {"action": action}


class RandomForwardAgent(RandomAgent):
    def __init__(self, success_distance: float, goal_sensor_uuid: str) -> None:
        super().__init__(success_distance, goal_sensor_uuid)
        self.FORWARD_PROBABILITY = 0.8

    def act(self, observations: Observations) -> Dict[str, Union[int, int64]]:
        if self.is_goal_reached(observations):
            action = HabitatSimActions.stop
        else:
            if np.random.uniform(0, 1, 1) < self.FORWARD_PROBABILITY:
                action = HabitatSimActions.move_forward
            else:
                action = np.random.choice(
                    [HabitatSimActions.turn_left, HabitatSimActions.turn_right]
                )

        return {"action": action}


class GoalFollower(RandomAgent):
    def __init__(self, success_distance: float, goal_sensor_uuid: str) -> None:
        super().__init__(success_distance, goal_sensor_uuid)
        self.pos_th = self.dist_threshold_to_stop
        self.angle_th = float(np.deg2rad(15))
        self.random_prob = 0

    def normalize_angle(self, angle: ndarray) -> ndarray:
        if angle < -pi:
            angle = 2.0 * pi + angle
        if angle > pi:
            angle = -2.0 * pi + angle
        return angle

    def turn_towards_goal(self, angle_to_goal: ndarray) -> int:
        if angle_to_goal > pi or (
            (angle_to_goal < 0) and (angle_to_goal > -pi)
        ):
            action = HabitatSimActions.turn_right
        else:
            action = HabitatSimActions.turn_left
        return action

    def act(self, observations: Observations) -> Dict[str, int]:
        if self.is_goal_reached(observations):
            action = HabitatSimActions.stop
        else:
            angle_to_goal = self.normalize_angle(
                np.array(observations[self.goal_sensor_uuid][1])
            )
            if abs(angle_to_goal) < self.angle_th:
                action = HabitatSimActions.move_forward
            else:
                action = self.turn_towards_goal(angle_to_goal)

        return {"action": action}


def get_all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in get_all_subclasses(c)]
    )


def get_agent_cls(agent_class_name):
    sub_classes = [
        sub_class
        for sub_class in get_all_subclasses(habitat.Agent)
        if sub_class.__name__ == agent_class_name
    ]
    return sub_classes[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--success-distance", type=float, default=0.2)
    parser.add_argument(
        "--task-config",
        type=str,
        default="habitat-lab/habitat/config/task/pointnav.yaml",
    )
    parser.add_argument("--agent-class", type=str, default="GoalFollower")
    args = parser.parse_args()

    config = get_config(args.task_config)

    agent = get_agent_cls(args.agent_class)(
        success_distance=args.success_distance,
        goal_sensor_uuid=config.habitat.task.goal_sensor_uuid,
    )
    benchmark = habitat.Benchmark(config_paths=args.task_config)
    metrics = benchmark.evaluate(agent)

    for k, v in metrics.items():
        habitat.logger.info("{}: {:.3f}".format(k, v))


if __name__ == "__main__":
    main()
