#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.datasets.rearrange.rearrange_dataset import RearrangeEpisode
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.tasks.rearrange.utils import (
    embodied_unoccluded_navmesh_snap,
    rearrange_logger,
    set_agent_base_via_obj_trans,
)


@registry.register_task(name="RearrangePickTask-v0")
class RearrangePickTaskV1(RearrangeTask):
    DISTANCE_TO_RECEPTACLE = 1.0
    """
    Rearrange Pick Task with Fetch robot interacting with objects and environment.
    """

    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(
            config=config,
            *args,
            dataset=dataset,
            should_place_articulated_agent=False,
            **kwargs,
        )

        self.prev_colls = None
        self.force_set_idx = None
        self._base_angle_noise = self._config.base_angle_noise
        self._spawn_max_dist_to_obj = self._config.spawn_max_dist_to_obj
        self._num_spawn_attempts = self._config.num_spawn_attempts
        self._filter_colliding_states = self._config.filter_colliding_states
        self._spawn_max_dist_to_obj_delta = (
            self._config.spawn_max_dist_to_obj_delta
        )

    def set_args(self, obj, **kwargs):
        self.force_set_idx = obj

    def _get_targ_pos(self, sim):
        scene_pos = sim.get_scene_pos()
        targ_idxs = sim.get_targets()[0]
        return scene_pos[targ_idxs]

    def _sample_idx(self, sim):
        if self.force_set_idx is not None:
            idxs = self._sim.get_targets()[0]
            sel_idx = self.force_set_idx
            sel_idx = list(idxs).index(sel_idx)
        else:
            sel_idx = np.random.randint(0, len(self._get_targ_pos(sim)))
        return sel_idx

    def _gen_start_pos(self, sim, episode, sel_idx):
        target_positions = self._get_targ_pos(sim)
        targ_pos = target_positions[sel_idx]

        was_fail = True
        spawn_attempt_count = 0

        while was_fail and spawn_attempt_count < self._num_spawn_attempts:
            (
                start_pos,
                angle_to_obj,
                success,
            ) = embodied_unoccluded_navmesh_snap(
                target_position=targ_pos,
                height=1.5,  # NOTE: this is default agent max height. This parameter is used to determine whether or not a point is occluded.
                sim=sim,
                island_id=sim._largest_indoor_island_idx,
                search_offset=self._spawn_max_dist_to_obj
                + spawn_attempt_count * self._spawn_max_dist_to_obj_delta,
                orientation_noise=self._base_angle_noise,
                max_samples=self._num_spawn_attempts,
                target_object_id=None,  # TODO: this must be the integer id of the target object or no unoccluded state will be found because this object will be considered occluding
                agent_embodiement=(
                    sim.articulated_agent
                    if self._filter_colliding_states
                    else None
                ),
            )
            spawn_attempt_count += 1

        if not success:
            rearrange_logger.error(
                f"Episode {episode.episode_id} failed to place robot"
            )

        return start_pos, angle_to_obj

    def _should_prevent_grip(self, action_args):
        return (
            self._sim.grasp_mgr.is_grasped
            and action_args.get("grip_action", None) is not None
            and action_args["grip_action"] < 0
        )

    def step(self, action, episode):
        action_args = action["action_args"]

        if self._should_prevent_grip(action_args):
            # No releasing the object once it is held.
            action_args["grip_action"] = None
        obs = super().step(action=action, episode=episode)

        return obs

    def reset(self, episode: Episode, fetch_observations: bool = True):
        sim = self._sim

        assert isinstance(
            episode, RearrangeEpisode
        ), "Provided episode needs to be of type RearrangeEpisode for RearrangePickTaskV1"

        super().reset(episode, fetch_observations=False)

        self.prev_colls = 0

        sel_idx = self._sample_idx(sim)
        start_pos, start_rot = self._gen_start_pos(sim, episode, sel_idx)

        set_agent_base_via_obj_trans(
            start_pos, start_rot, sim.articulated_agent
        )

        self._targ_idx = sel_idx

        if fetch_observations:
            self._sim.maybe_update_articulated_agent()
            return self._get_observations(episode)
        return None
