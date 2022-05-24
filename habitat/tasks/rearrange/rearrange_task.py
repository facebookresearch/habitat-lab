#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Any, Dict, List, Union

import numpy as np

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.nav.nav import NavigationTask
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.utils import (
    CollisionDetails,
    rearrange_collision,
    rearrange_logger,
)


def merge_sim_episode_with_object_config(sim_config, episode):
    sim_config.defrost()
    sim_config.ep_info = [episode.__dict__]
    sim_config.freeze()
    return sim_config


ADD_CACHE_KEY = "add_cache_key"


@registry.register_task(name="RearrangeEmptyTask-v0")
class RearrangeTask(NavigationTask):
    """
    Defines additional logic for valid collisions and gripping shared between
    all rearrangement tasks.
    """

    _cur_episode_step: int

    def overwrite_sim_config(self, sim_config, episode):
        return merge_sim_episode_with_object_config(sim_config, episode)

    def __init__(self, *args, sim, dataset=None, **kwargs) -> None:
        self.n_objs = len(dataset.episodes[0].targets)

        super().__init__(sim=sim, dataset=dataset, **kwargs)
        self.is_gripper_closed = False
        self._sim: RearrangeSim = sim
        self._ignore_collisions: List[Any] = []
        self._desired_resting = np.array(self._config.DESIRED_RESTING_POSITION)
        self._sim_reset = True
        self._targ_idx: int = 0
        self._episode_id: str = ""
        self._cur_episode_step = 0

    @property
    def targ_idx(self):
        return self._targ_idx

    @property
    def abs_targ_idx(self):
        if self._targ_idx is None:
            return None
        return self._sim.get_targets()[0][self._targ_idx]

    @property
    def desired_resting(self):
        return self._desired_resting

    def set_args(self, **kwargs):
        raise NotImplementedError("Task cannot dynamically set arguments")

    def set_sim_reset(self, sim_reset):
        self._sim_reset = sim_reset

    def reset(self, episode: Episode, fetch_observations: bool = True):
        self._episode_id = episode.episode_id
        self._ignore_collisions = []

        if self._sim_reset:
            self._sim.reset()
            for action_instance in self.actions.values():
                action_instance.reset(episode=episode, task=self)
            self._is_episode_active = True
            self._sim.set_robot_base_to_random_point()

        self.prev_measures = self.measurements.get_metrics()
        self._targ_idx = 0
        self.coll_accum = CollisionDetails()
        self.prev_coll_accum = CollisionDetails()
        self.should_end = False
        self._done = False
        self._cur_episode_step = 0
        if fetch_observations:
            return self._get_observations(episode)
        else:
            return None

    def _get_observations(self, episode):
        obs = self._sim.get_sensor_observations()
        obs = self._sim._sensor_suite.get_observations(obs)

        task_obs = self.sensor_suite.get_observations(
            observations=obs, episode=episode, task=self
        )
        obs.update(task_obs)
        return obs

    def step(self, action: Dict[str, Any], episode: Episode):
        obs = super().step(action=action, episode=episode)

        self.prev_coll_accum = copy.copy(self.coll_accum)
        self._cur_episode_step += 1

        return obs

    def _check_episode_is_active(
        self,
        *args: Any,
        action: Union[int, Dict[str, Any]],
        episode: Episode,
        **kwargs: Any,
    ) -> bool:
        done = False
        if self.should_end:
            done = True

        if (
            self._sim.grasp_mgr.is_violating_hold_constraint()
            and self._config.CONSTRAINT_VIOLATION_ENDS_EPISODE
        ):
            done = True

        if done:
            rearrange_logger.debug("-" * 10)
            rearrange_logger.debug("------ Episode Over --------")
            rearrange_logger.debug("-" * 10)

        return not done

    def get_coll_forces(self):
        snapped_obj = self._sim.grasp_mgr.snap_idx
        robot_id = self._sim.robot.sim_obj.object_id
        contact_points = self._sim.get_physics_contact_points()

        def get_max_force(contact_points, check_id):
            match_contacts = [
                x
                for x in contact_points
                if check_id in [x.object_id_a, x.object_id_b]
            ]
            match_contacts = [
                x for x in match_contacts if x.object_id_a != x.object_id_b
            ]

            max_force = 0
            if len(match_contacts) > 0:
                max_force = max([abs(x.normal_force) for x in match_contacts])

            return max_force

        forces = [
            abs(x.normal_force)
            for x in contact_points
            if (
                x.object_id_a not in self._ignore_collisions
                and x.object_id_b not in self._ignore_collisions
            )
        ]
        max_force = max(forces) if len(forces) > 0 else 0

        max_obj_force = get_max_force(contact_points, snapped_obj)
        max_robot_force = get_max_force(contact_points, robot_id)
        return max_robot_force, max_obj_force, max_force

    def get_cur_collision_info(self) -> CollisionDetails:
        _, coll_details = rearrange_collision(
            self._sim,
            self._config.COUNT_OBJ_COLLISIONS,
        )
        return coll_details

    def get_n_targets(self) -> int:
        return self.n_objs

    @property
    def should_end(self) -> bool:
        return self._should_end

    @should_end.setter
    def should_end(self, new_val: bool):
        self._should_end = new_val
        ##
        # NB: _check_episode_is_active is called after step() but
        # before metrics are updated. Thus if should_end is set
        # by a metric, the episode will end on the _next_
        # step. This makes sure that the episode is ended
        # on the correct step.
        self._is_episode_active = (
            not self._should_end
        ) and self._is_episode_active
        if new_val:
            rearrange_logger.debug("-" * 40)
            rearrange_logger.debug(
                f"-----Episode {self._episode_id} requested to end after {self._cur_episode_step} steps.-----"
            )
            rearrange_logger.debug("-" * 40)
