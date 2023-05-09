#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.datasets.rearrange.rearrange_dataset import RearrangeEpisode
from habitat.robots.stretch_robot import StretchRobot
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.tasks.rearrange.utils import get_robot_spawns, rearrange_logger


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
            should_place_robot=False,
            **kwargs,
        )
        self._spawn_reference = config.spawn_reference
        self._spawn_reference_sampling = config.spawn_reference_sampling
        self.prev_colls = None
        self.force_set_idx = None

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

    def _get_spawn_goals(self, episode):
        if self._spawn_reference == 'view_points':
            return episode.candidate_objects
        else:
            return episode.candidate_start_receps
    

    def get_spawn_reference_points(self, sim, episode, sel_idx):
        # Return a tuple of numpy arrays, the first being the reference points for distance and the second being the reference points for angle 
        if self._spawn_reference == 'receptacle_center':
            assert self._spawn_reference_sampling == 'uniform', 'Only uniform sampling is supported for receptacle center'
            recep_centers = np.array([g.position for g in self._get_spawn_goals(episode)])
            return recep_centers, recep_centers, None
        elif self._spawn_reference == 'target':
            assert self._spawn_reference_sampling == 'uniform', 'Only uniform sampling is supported for target'
            # biased init wrt geogoal pick or place target
            target_positions = self._get_targ_pos(sim)
            target_positions = np.expand_dims(target_positions[sel_idx], axis=0)
            return target_positions, target_positions, None
        elif self._spawn_reference == 'view_points':
            recep_view_points = np.array([v.agent_state.position for g in self._get_spawn_goals(episode) for v in g.view_points])
            recep_centers = np.array([g.position for g in self._get_spawn_goals(episode) for _ in g.view_points])
            
            if self._spawn_reference_sampling == 'uniform':
                return recep_view_points, recep_centers, None
            elif self._spawn_reference_sampling == 'dist_to_center':
                # TODO: use distance to the edge or cache the distances
                dist_to_recep_center = [np.linalg.norm(np.array([v.agent_state.position for v in g.view_points]) - g.position, axis=1) for g in self._get_spawn_goals(episode)]
                normalized_dist_to_center = [dists_per_recep / np.sum(dists_per_recep) for dists_per_recep in dist_to_recep_center]
                sample_probs = [d for dists_per_recep in normalized_dist_to_center for d in dists_per_recep]
                return recep_view_points, recep_centers, sample_probs / np.sum(sample_probs)
            else:
                raise ValueError(f"Unrecognized spawn reference sampling {self._spawn_reference_sampling}")
        else:
            raise ValueError(f"Unrecognized spawn reference {self._spawn_reference}") 



    def _gen_start_pos(self, sim, episode, sel_idx):
        snap_pos, orient_pos, sample_probs = self.get_spawn_reference_points(sim, episode, sel_idx)
        start_pos, angle_to_obj, was_succ = get_robot_spawns(
            snap_pos,
            self._config.base_angle_noise,
            self._config.spawn_max_dists_to_obj,
            sim,
            self._config.num_spawn_attempts,
            self._config.physics_stability_steps,
            orient_positions=orient_pos,
            sample_probs=sample_probs,
        )

        if was_succ:
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
        # in the case of Stretch, force the agent to look down and retract arm with the gripper pointing downwards
        if isinstance(sim.robot, StretchRobot):
            sim.robot.arm_motor_pos = np.array(
                [0.0] * 4 + [0.775, 0.0, -1.57000005, 0.0, -1.7375, -0.7125]
            )
            sim.robot.arm_joint_pos = np.array(
                [0.0] * 4 + [0.775, 0.0, -1.57000005, 0.0, -1.7375, -0.7125]
            )
        start_pos, start_rot = self._gen_start_pos(sim, episode, sel_idx)

        sim.robot.base_pos = start_pos
        # in the case of Stretch, rotate base so that the arm faces the target location
        if isinstance(self._sim.robot, StretchRobot):
            sim.robot.base_rot = start_rot + np.pi / 2
        else:
            sim.robot.base_rot = start_rot

        self._targ_idx = sel_idx

        if fetch_observations:
            self._sim.maybe_update_robot()
            return self._get_observations(episode)
        return None
