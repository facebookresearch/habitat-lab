#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp

import numpy as np

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.tasks.rearrange.utils import CacheHelper, rearrang_collision
from habitat.tasks.utils import get_angle


def merge_sim_episode_with_object_config(sim_config, episode):
    sim_config.defrost()
    sim_config.ep_info = [episode.__dict__]
    sim_config.freeze()
    return sim_config


@registry.register_task(name="RearrangePickTask-v0")
class RearrangePickTaskV1(RearrangeTask):

    """
    Rearrange Pick Task with Fetch robot interacting with objects and environment.
    """

    def overwrite_sim_config(self, sim_config, episode):
        return merge_sim_episode_with_object_config(sim_config, episode)

    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(config=config, *args, dataset=dataset, **kwargs)
        data_path = dataset.config.DATA_PATH.format(split=dataset.config.SPLIT)

        mtime = osp.getmtime(data_path)
        cache_name = (
            str(mtime)
            + dataset.config.SPLIT
            + str(self._config.COUNT_OBJ_COLLISIONS)
        )
        cache_name += str(self._config.BASE_NOISE)
        cache_name = cache_name.replace(".", "_")

        fname = data_path.split("/")[-1].split(".")[0]

        self.cache = CacheHelper(
            "start_pos", cache_name, {}, verbose=False, rel_dir=fname
        )
        self.start_states = self.cache.load()
        self.desired_resting = np.array([0.5, 0.0, 1.0])
        self.targ_idx = None
        self.force_set_idx = None
        self.prev_colls = None
        self.abs_targ_idx = None
        self.cur_dist = -1.0
        self.prev_picked = False

    def _is_holding_obj(self):
        return self._sim.snapped_obj_id is not None

    def _get_targ_pos(self, sim):
        return sim.get_target_objs_start()

    def _gen_start_pos(self, sim, is_easy_init):
        target_positions = self._get_targ_pos(sim)
        sel_idx = np.random.randint(0, len(target_positions))
        if self.force_set_idx is not None:
            sel_idx = self.force_set_idx
        targ_pos = target_positions[sel_idx]

        orig_start_pos = sim.get_nav_pos(targ_pos, True)

        state = sim.capture_state()
        start_pos = orig_start_pos

        forward = np.array([1.0, 0, 0])
        dist_thresh = 0.1
        did_collide = False

        # Add a bit of noise
        timeout = 1000
        for _attempt in range(timeout):
            start_pos = orig_start_pos + np.random.normal(
                0, self._config.BASE_NOISE, size=(3,)
            )
            targ_dist = np.linalg.norm((start_pos - orig_start_pos)[[0, 2]])

            is_navigable = is_easy_init or sim.pathfinder.is_navigable(
                start_pos
            )

            if targ_dist > dist_thresh or not is_navigable:
                continue

            sim.set_state(state)

            sim.set_robot_pos(start_pos[[0, 2]])

            # Face the robot towards the object.
            rel_targ = targ_pos - start_pos
            angle_to_obj = get_angle(forward[[0, 2]], rel_targ[[0, 2]])
            if np.cross(forward[[0, 2]], rel_targ[[0, 2]]) > 0:
                angle_to_obj *= -1.0
            sim.set_robot_rot(angle_to_obj)

            # Make sure the robot is not colliding with anything in this
            # position.
            for _ in range(100):
                sim.internal_step(-1)
                colls = sim.get_collisions()
                did_collide, details = rearrang_collision(
                    colls,
                    None,
                    self._config.COUNT_OBJ_COLLISIONS,
                    ignore_base=False,
                )

                if is_easy_init:
                    # Only care about collisions between the robot and scene.
                    did_collide = details.robo_scene_colls != 0

                if did_collide:
                    break

            if not did_collide:
                break

        if _attempt == timeout - 1 and (not is_easy_init):
            start_pos, angle_to_obj, sel_idx = self._gen_start_pos(sim, True)

        sim.set_state(state)

        return start_pos, angle_to_obj, sel_idx

    def _should_prevent_grip(self, action_args):
        return (
            self._is_holding_obj()
            and action_args.get("grip_ac", None) is not None
            and action_args["grip_ac"] <= 0
        )

    def step(self, action, episode):
        action_args = action["action_args"]

        if self._should_prevent_grip(action_args):
            # No releasing the object once it is held.
            action_args["grip_ac"] = None
        obs = super().step(action=action, episode=episode)

        return obs

    def set_args(self, obj, **kwargs):
        self.force_set_idx = obj

    def reset(self, episode: Episode):
        sim = self._sim

        super().reset(episode)

        self.prev_colls = 0
        episode_id = sim.ep_info["episode_id"]

        if episode_id in self.start_states and self.force_set_idx is None:
            start_pos, start_rot, sel_idx = self.start_states[episode_id]
        else:
            start_pos, start_rot, sel_idx = self._gen_start_pos(
                sim, self._config.EASY_INIT
            )
            self.start_states[episode_id] = (start_pos, start_rot, sel_idx)
            if self.force_set_idx is None:
                self.cache.save(self.start_states)

        for _ in range(5):
            sim.internal_step(-1)
            colls = sim.get_collisions()
            did_collide, _ = rearrang_collision(
                colls,
                None,
                self._config.COUNT_OBJ_COLLISIONS,
                ignore_base=False,
            )
            rot_noise = np.random.normal(0.0, self._config.BASE_ANGLE_NOISE)

            sim.set_robot_pos(start_pos[[0, 2]])
            sim.set_robot_rot(start_rot + rot_noise)
            if not did_collide:
                break

        self.targ_idx = sel_idx
        assert self.targ_idx is not None
        self.abs_targ_idx = sim.get_targets()[0][sel_idx]
        # Value < 0 will not be used
        self.cur_dist = -1.0
        snapped_id = self._sim.snapped_obj_id
        self.prev_picked = snapped_id is not None

        return super(RearrangePickTaskV1, self).reset(episode)
