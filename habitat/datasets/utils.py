#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from habitat.core.logging import logger
from habitat.core.simulator import ShortestPathPoint, SimulatorActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.geometry_utils import quaternion_to_list


def get_action_shortest_path(
    sim,
    source_position,
    source_rotation,
    goal_position,
    success_distance=0.05,
    max_episode_steps=500,
    shortest_path_mode="greedy",
) -> List[ShortestPathPoint]:
    sim.reset()
    sim.set_agent_state(source_position, source_rotation)
    follower = ShortestPathFollower(sim, success_distance, False)
    follower.mode = shortest_path_mode

    shortest_path = []
    step_count = 0
    action = follower.get_next_action(goal_position)
    while action is not None and step_count < max_episode_steps:
        state = sim.get_agent_state()
        shortest_path.append(
            ShortestPathPoint(
                state.position.tolist(),
                quaternion_to_list(state.rotation),
                action,
            )
        )
        sim.step(action)
        step_count += 1
        action = follower.get_next_action(goal_position)

    if step_count == max_episode_steps:
        logger.warning("Shortest path wasn't found.")
    return shortest_path
