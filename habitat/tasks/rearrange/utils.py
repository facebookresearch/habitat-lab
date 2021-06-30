#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import os
import os.path as osp
import pickle
import time

import attr
import gym
import magnum as mn
import numpy as np
import quaternion

import habitat_sim
from habitat_sim.nav import NavMeshSettings
from habitat_sim.physics import MotionType


def make_render_only(obj_idx, sim):
    if hasattr(MotionType, "RENDER_ONLY"):
        sim.set_object_motion_type(MotionType.RENDER_ONLY, obj_idx)
    else:
        sim.set_object_motion_type(MotionType.KINEMATIC, obj_idx)
        sim.set_object_is_collidable(False, obj_idx)


def get_collision_matches(link, colls, search_key="link"):
    matches = []
    for coll0, coll1 in colls:
        if link in [coll0[search_key], coll1[search_key]]:
            matches.append((coll0, coll1))
    return matches


def get_other_matches(link, colls):
    matches = get_collision_matches(link, colls)
    other_surfaces = [b if a["link"] == link else a for a, b in matches]
    return other_surfaces


def coll_name(coll, name):
    return coll_prop(coll, name, "name")


def coll_prop(coll, val, prop):
    return val in [coll[0][prop], coll[1][prop]]


def coll_link(coll, link):
    return coll_prop(coll, link, "link")


def swap_axes(x):
    x[1], x[2] = x[2], x[1]
    return x


@attr.s(auto_attribs=True, kw_only=True)
class CollDetails:
    obj_scene_colls: int = 0
    robo_obj_colls: int = 0
    robo_scene_colls: int = 0


def rearrang_collision(
    colls,
    snapped_obj_id,
    count_obj_colls,
    verbose=False,
    ignore_names=None,
    ignore_base=True,
):
    """
    Defines what counts as a collision for the Rearrange environment execution
    """
    # Filter out any collisions from the base
    if ignore_base:
        colls = [
            x
            for x in colls
            if not ("base" in x[0]["link"] or "base" in x[1]["link"])
        ]

    def should_keep(x):
        if ignore_names is None:
            return True
        return any(coll_name(x, ignore_name) for ignore_name in ignore_names)

    # Filter out any collisions with the ignore objects
    colls = list(filter(should_keep, colls))

    # Check for robot collision
    robo_obj_colls = 0
    robo_scene_colls = 0
    robo_scene_matches = get_collision_matches("fetch", colls, "name")
    for match in robo_scene_matches:
        urdf_on_urdf = (
            match[0]["type"] == "URDF" and match[1]["type"] == "URDF"
        )
        with_stage = coll_prop(match, "Stage", "type")
        fetch_on_fetch = (
            match[0]["name"] == "fetch" and match[1]["name"] == "fetch"
        )
        if fetch_on_fetch:
            continue
        if urdf_on_urdf or with_stage:
            robo_scene_colls += 1
        else:
            robo_obj_colls += 1

    # Checking for holding object collision
    obj_scene_colls = 0
    if count_obj_colls and snapped_obj_id is not None:
        matches = get_collision_matches(
            "id %i" % snapped_obj_id, colls, "link"
        )
        for match in matches:
            if coll_name(match, "fetch"):
                continue
            obj_scene_colls += 1

    total_colls = robo_obj_colls + robo_scene_colls + obj_scene_colls
    return total_colls > 0, CollDetails(
        obj_scene_colls=min(obj_scene_colls, 1),
        robo_obj_colls=min(robo_obj_colls, 1),
        robo_scene_colls=min(robo_scene_colls, 1),
    )


def get_nav_mesh_settings(agent_config):
    return get_nav_mesh_settings_from_height(agent_config.HEIGHT)


def get_nav_mesh_settings_from_height(height):
    navmesh_settings = NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.agent_radius = 0.4
    navmesh_settings.agent_height = height
    navmesh_settings.agent_max_climb = 0.05
    return navmesh_settings


def convert_legacy_cfg(obj_list):
    if len(obj_list) == 0:
        return obj_list

    def convert_fn(obj_dat):
        fname = "/".join(obj_dat[0].split("/")[-2:])
        if ".urdf" in fname:
            obj_dat[0] = osp.join("data/replica_cad/urdf", fname)
        else:
            obj_dat[0] = obj_dat[0].replace(
                "data/objects/", "data/objects/ycb/"
            )

        if (
            len(obj_dat) == 2
            and len(obj_dat[1]) == 4
            and np.array(obj_dat[1]).shape == (4, 4)
        ):
            # Specifies the full transformation, no object type
            return (obj_dat[0], (obj_dat[1], int(MotionType.DYNAMIC)))
        elif len(obj_dat) == 2 and len(obj_dat[1]) == 3:
            # Specifies XYZ, no object type
            trans = mn.Matrix4.translation(mn.Vector3(obj_dat[1]))
            return (obj_dat[0], (trans, int(MotionType.DYNAMIC)))
        else:
            # Specifies the full transformation and the object type
            return (obj_dat[0], obj_dat[1])

    return list(map(convert_fn, obj_list))


def get_aabb(obj_id, sim, transformed=False):
    obj_node = sim.get_object_scene_node(obj_id)
    obj_bb = obj_node.cumulative_bb
    if transformed:
        obj_bb = habitat_sim.geo.get_transformed_bb(
            obj_node.cumulative_bb, obj_node.transformation
        )
    return obj_bb


def euler_to_quat(rpy):
    rot = quaternion.from_euler_angles(rpy)
    rot = mn.Quaternion(mn.Vector3(rot.vec), rot.w)
    return rot


def allowed_region_to_bb(allowed_region):
    if len(allowed_region) == 0:
        return allowed_region
    return mn.Range2D(allowed_region[0], allowed_region[1])


def recover_nav_island_point(v, ref_v, sim):
    """
    Snaps a point to the LARGEST island.
    """
    nav_vs = sim.pathfinder.build_navmesh_vertices()
    ref_r = sim.pathfinder.island_radius(ref_v)

    nav_vs_r = {
        i: sim.pathfinder.island_radius(nav_v)
        for i, nav_v in enumerate(nav_vs)
    }
    # Get the points closest to "v"
    v_dist = np.linalg.norm(v - nav_vs, axis=-1)
    ordered_idxs = np.argsort(v_dist)

    # Go through the closest points until one has the same island radius.
    for i in ordered_idxs:
        if nav_vs_r[i] == ref_r:
            return nav_vs[i]
    print("Could not find point off of island")
    return v


CACHE_PATH = "./data/cache"


class CacheHelper:
    def __init__(
        self, cache_name, lookup_val, def_val=None, verbose=False, rel_dir=""
    ):
        self.use_cache_path = osp.join(CACHE_PATH, rel_dir)
        if not osp.exists(self.use_cache_path):
            os.makedirs(self.use_cache_path)
        sec_hash = hashlib.md5(str(lookup_val).encode("utf-8")).hexdigest()
        cache_id = f"{cache_name}_{sec_hash}.pickle"
        self.cache_id = osp.join(self.use_cache_path, cache_id)
        self.def_val = def_val
        self.verbose = verbose

    def exists(self):
        return osp.exists(self.cache_id)

    def load(self, load_depth=0):
        if not self.exists():
            return self.def_val
        try:
            with open(self.cache_id, "rb") as f:
                if self.verbose:
                    print("Loading cache @", self.cache_id)
                return pickle.load(f)
        except EOFError as e:
            if load_depth == 32:
                raise e
            # try again soon
            print(
                "Cache size is ",
                osp.getsize(self.cache_id),
                "for ",
                self.cache_id,
            )
            time.sleep(1.0 + np.random.uniform(0.0, 1.0))
            return self.load(load_depth + 1)

    def save(self, val):
        with open(self.cache_id, "wb") as f:
            if self.verbose:
                print("Saving cache @", self.cache_id)
            pickle.dump(val, f)


def reshape_obs_space(obs_space, new_shape):
    assert isinstance(obs_space, gym.spaces.Box)
    return gym.spaces.Box(
        shape=new_shape,
        high=obs_space.low.reshape(-1)[0],
        low=obs_space.high.reshape(-1)[0],
        dtype=obs_space.dtype,
    )
