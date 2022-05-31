#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
from typing import Tuple

import magnum as mn
import numpy as np

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.rearrange.rearrange_task import ADD_CACHE_KEY, RearrangeTask
from habitat.tasks.rearrange.utils import (
    CacheHelper,
    rearrange_collision,
    rearrange_logger,
)
from habitat.tasks.utils import get_angle


@registry.register_task(name="RearrangePickTask-v0")
class RearrangePickTaskV1(RearrangeTask):
    DISTANCE_TO_RECEPTACLE = 1.0
    """
    Rearrange Pick Task with Fetch robot interacting with objects and environment.
    """

    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(config=config, *args, dataset=dataset, **kwargs)
        data_path = dataset.config.DATA_PATH.format(split=dataset.config.SPLIT)

        fname = data_path.split("/")[-1].split(".")[0]
        save_dir = osp.dirname(data_path)
        self.cache = CacheHelper(
            osp.join(save_dir, f"{fname}_{config.TYPE}_start.pickle"),
            def_val={},
            verbose=False,
        )
        self.start_states = self.cache.load()
        self.prev_colls = None
        self.force_set_idx = None
        self._add_cache_key: str = ""

    def set_args(self, obj, **kwargs):
        self.force_set_idx = obj
        if ADD_CACHE_KEY in kwargs:
            self._add_cache_key = kwargs[ADD_CACHE_KEY]

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

    @property
    def _is_there_spawn_noise(self):
        return (
            self._config.BASE_NOISE != 0.0
            or self._config.BASE_ANGLE_NOISE != 0
        )

    def _gen_start_pos(
        self, sim, is_easy_init, episode, sel_idx, force_snap_pos=None
    ):
        target_positions = self._get_targ_pos(sim)
        targ_pos = target_positions[sel_idx]

        if force_snap_pos is not None:
            snap_pos = force_snap_pos
        else:
            snap_pos = targ_pos

        orig_start_pos = sim.safe_snap_point(snap_pos)

        state = sim.capture_state()
        start_pos = orig_start_pos

        forward = np.array([1.0, 0, 0])
        dist_thresh = 0.1
        did_collide = False

        if self._config.SHOULD_ENFORCE_TARGET_WITHIN_REACH:
            # Setting so the object is within reach is harder and requires more
            # tries.
            timeout = 5000
        else:
            timeout = 1000
        attempt = 0
        is_within_bounds = True

        # Add noise to the base position and angle for a collision free
        # starting position
        while attempt < timeout:
            attempt += 1
            start_pos = orig_start_pos + np.random.normal(
                0, self._config.BASE_NOISE, size=(3,)
            )
            rel_targ = targ_pos - start_pos
            angle_to_obj = get_angle(forward[[0, 2]], rel_targ[[0, 2]])
            if np.cross(forward[[0, 2]], rel_targ[[0, 2]]) > 0:
                angle_to_obj *= -1.0

            if not self._is_there_spawn_noise:
                rearrange_logger.debug(
                    "No spawn noise, returning first found position"
                )
                break

            targ_dist = np.linalg.norm((start_pos - orig_start_pos)[[0, 2]])

            is_navigable = is_easy_init or sim.pathfinder.is_navigable(
                start_pos
            )

            if targ_dist > dist_thresh or not is_navigable:
                continue

            sim.set_state(state)

            sim.robot.base_pos = start_pos

            # Face the robot towards the object.
            rot_noise = np.random.normal(0.0, self._config.BASE_ANGLE_NOISE)
            sim.robot.base_rot = angle_to_obj + rot_noise

            # Ensure the target is within reach
            is_within_bounds = True
            if self._config.SHOULD_ENFORCE_TARGET_WITHIN_REACH:
                robot_T = self._sim.robot.base_transformation
                rel_targ_pos = robot_T.inverted().transform_point(targ_pos)
                eps = 1e-2
                upper_bound = self._sim.robot.params.ee_constraint[:, 1] + eps
                is_within_bounds = (rel_targ_pos < upper_bound).all()
                if not is_within_bounds:
                    continue

            # Make sure the robot is not colliding with anything in this
            # position.
            for _ in range(100):
                sim.internal_step(-1)
                did_collide, details = rearrange_collision(
                    self._sim,
                    self._config.COUNT_OBJ_COLLISIONS,
                    ignore_base=False,
                )

                if is_easy_init:
                    # Only care about collisions between the robot and scene.
                    did_collide = details.robot_scene_colls != 0

                if did_collide:
                    break

            if not did_collide:
                break

        if attempt == timeout and (not is_easy_init):
            start_pos, angle_to_obj = self._gen_start_pos(
                sim, True, episode, sel_idx
            )
        elif not is_within_bounds or attempt == timeout or did_collide:
            rearrange_logger.error(
                f"Episode {episode.episode_id} failed to place robot"
            )

        sim.set_state(state)

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

    def get_receptacle_info(
        self, episode: Episode, sel_idx: int
    ) -> Tuple[str, int]:
        """
        Returns the receptacle handle and receptacle parent link index.
        """
        return episode.target_receptacles[sel_idx]

    def reset(self, episode: Episode, fetch_observations: bool = True):
        sim = self._sim

        super().reset(episode, fetch_observations=False)

        self.prev_colls = 0
        cache_lookup_k = sim.ep_info["episode_id"]
        cache_lookup_k += self._add_cache_key

        if self.force_set_idx is not None:
            cache_lookup_k += str(self.force_set_idx)
        rearrange_logger.debug(
            f"Using cache key {cache_lookup_k}, force_regenerate={self._config.FORCE_REGENERATE}"
        )

        if (
            cache_lookup_k in self.start_states
            and not self._config.FORCE_REGENERATE
        ):
            start_pos, start_rot, sel_idx = self.start_states[cache_lookup_k]
        else:
            mgr = sim.get_articulated_object_manager()
            sel_idx = self._sample_idx(sim)

            rearrange_logger.debug(
                f"Generating init for {self} and force set idx {self.force_set_idx} with selected object idx {sel_idx}"
            )

            receptacle_handle, receptacle_link_idx = self.get_receptacle_info(
                episode, sel_idx
            )
            if (
                # Not a typo, "fridge" is sometimes "frige" in
                # ReplicaCAD.
                receptacle_handle is not None
                and (
                    "frige" in receptacle_handle
                    or "fridge" in receptacle_handle
                )
            ):
                receptacle_ao = mgr.get_object_by_handle(receptacle_handle)
                start_pos = np.array(
                    receptacle_ao.transformation.transform_point(
                        mn.Vector3(self.DISTANCE_TO_RECEPTACLE, 0, 0)
                    )
                )
            elif (
                receptacle_handle is not None
                and "kitchen_counter" in receptacle_handle
                and receptacle_link_idx != 0
            ):
                receptacle_ao = mgr.get_object_by_handle(receptacle_handle)
                link_T = receptacle_ao.get_link_scene_node(
                    receptacle_link_idx
                ).transformation
                start_pos = np.array(
                    link_T.transform_point(mn.Vector3(0.8, 0, 0))
                )
            else:
                start_pos = None

            start_pos, start_rot = self._gen_start_pos(
                sim, self._config.EASY_INIT, episode, sel_idx, start_pos
            )
            rearrange_logger.debug(f"Finished creating init for {self}")
            self.start_states[cache_lookup_k] = (start_pos, start_rot, sel_idx)
            if self._config.SHOULD_SAVE_TO_CACHE:
                self.cache.save(self.start_states)

        sim.robot.base_pos = start_pos
        sim.robot.base_rot = start_rot

        self._targ_idx = sel_idx

        if fetch_observations:
            return self._get_observations(episode)
        return None
