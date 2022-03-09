#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp

import magnum as mn
import numpy as np

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.tasks.rearrange.utils import CacheHelper, rearrange_collision
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

        mtime = osp.getmtime(data_path)
        cache_name = str(mtime) + dataset.config.SPLIT
        cache_name += str(self._config.BASE_NOISE)
        cache_name = cache_name.replace(".", "_")

        fname = data_path.split("/")[-1].split(".")[0]

        self.cache = CacheHelper(
            "start_pos", cache_name, {}, verbose=False, rel_dir=fname
        )
        self.start_states = {}  # self.cache.load()
        self.prev_colls = None
        self.force_set_idx = None

    def set_args(self, obj, **kwargs):
        self.force_set_idx = obj

    def _get_targ_pos(self, sim):
        return sim.get_target_objs_start()

    def _gen_start_pos(self, sim, is_easy_init, episode, force_snap_pos=None):
        target_positions = self._get_targ_pos(sim)
        if self.force_set_idx is not None:
            sel_idx = self.force_set_idx
        else:
            sel_idx = np.random.randint(0, len(target_positions))
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
            start_pos, angle_to_obj, sel_idx = self._gen_start_pos(
                sim, True, episode
            )
        elif not is_within_bounds or attempt == timeout or did_collide:
            print(f"Episode {episode.episode_id} failed to place robot")
            print("Rel targ pos", rel_targ_pos, upper_bound)
            print("is within", is_within_bounds)
            print("Did collide", did_collide)

        sim.set_state(state)

        return start_pos, angle_to_obj, sel_idx

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

    def reset(self, episode: Episode):
        sim = self._sim

        super().reset(episode)

        self.prev_colls = 0
        episode_id = sim.ep_info["episode_id"]

        if (
            episode_id in self.start_states
            and not self._config.FORCE_REGENERATE
        ):
            start_pos, start_rot, sel_idx = self.start_states[episode_id]
        else:
            mgr = sim.get_articulated_object_manager()
            start_pos = None
            if (
                len(episode.targets.keys()) == 1
                and episode.target_receptacles is not None
            ):
                receptacle_handle = episode.target_receptacles[0]
                receptacle_link_idx = episode.target_receptacles[1]
                if (
                    # Not a typo, "fridge" is sometimes "frige" in
                    # ReplicaCAD.
                    "frige" in receptacle_handle
                    or "fridge" in receptacle_handle
                ):
                    receptacle_ao = mgr.get_object_by_handle(receptacle_handle)
                    start_pos = np.array(
                        receptacle_ao.transformation.transform_point(
                            mn.Vector3(self.DISTANCE_TO_RECEPTACLE, 0, 0)
                        )
                    )

                if (
                    "kitchen_counter" in receptacle_handle
                    and receptacle_link_idx != 0
                ):
                    receptacle_ao = mgr.get_object_by_handle(receptacle_handle)
                    link_T = receptacle_ao.get_link_scene_node(
                        receptacle_link_idx
                    ).transformation
                    start_pos = np.array(
                        link_T.transform_point(mn.Vector3(0.8, 0, 0))
                    )

            start_pos, start_rot, sel_idx = self._gen_start_pos(
                sim, self._config.EASY_INIT, episode, start_pos
            )
            self.start_states[episode_id] = (start_pos, start_rot, sel_idx)
            self.cache.save(self.start_states)

        sim.robot.base_pos = start_pos
        sim.robot.base_rot = start_rot

        self._targ_idx = sel_idx

        return self._get_observations(episode)
