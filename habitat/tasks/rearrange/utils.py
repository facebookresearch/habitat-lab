#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import os
import os.path as osp
import pickle
import time
from typing import List, Optional, Tuple

import attr
import gym
import magnum as mn
import numpy as np
import quaternion

import habitat_sim
from habitat_sim.nav import NavMeshSettings
from habitat_sim.physics import MotionType


def make_render_only(obj, sim):
    obj.motion_type = MotionType.KINEMATIC
    obj.collidable = False


def make_border_red(img):
    border_color = [255, 0, 0]
    border_width = 10
    img[:, :border_width] = border_color
    img[:border_width, :] = border_color
    img[-border_width:, :] = border_color
    img[:, -border_width:] = border_color
    return img


def coll_name_matches(coll, name):
    return name in [coll.object_id_a, coll.object_id_b]


def coll_link_name_matches(coll, name):
    return name in [coll.link_id_a, coll.link_id_b]


def get_match_link(coll, name):
    if name == coll.object_id_a:
        return coll.link_id_a
    if name == coll.object_id_b:
        return coll.link_id_b
    return None


def swap_axes(x):
    x[1], x[2] = x[2], x[1]
    return x


@attr.s(auto_attribs=True, kw_only=True)
class CollisionDetails:
    obj_scene_colls: int = 0
    robot_obj_colls: int = 0
    robot_scene_colls: int = 0
    robot_coll_ids: List[int] = []
    all_colls: List[Tuple[int, int]] = []

    @property
    def total_collisions(self):
        return (
            self.obj_scene_colls
            + self.robot_obj_colls
            + self.robot_scene_colls
        )

    def __add__(self, other):
        return CollisionDetails(
            obj_scene_colls=self.obj_scene_colls + other.obj_scene_colls,
            robot_obj_colls=self.robot_obj_colls + other.robot_obj_colls,
            robot_scene_colls=self.robot_scene_colls + other.robot_scene_colls,
            robot_coll_ids=[*self.robot_coll_ids, *other.robot_coll_ids],
            all_colls=[*self.all_colls, *other.all_colls],
        )


def rearrange_collision(
    sim,
    count_obj_colls: bool,
    verbose: bool = False,
    ignore_names: Optional[List[str]] = None,
    ignore_base: bool = True,
    get_extra_coll_data: bool = False,
):
    """Defines what counts as a collision for the Rearrange environment execution"""
    robot_model = sim.robot
    colls = sim.get_physics_contact_points()
    robot_id = robot_model.get_robot_sim_id()
    added_objs = sim.scene_obj_ids
    snapped_obj_id = sim.grasp_mgr.snap_idx

    def should_keep(x):
        if ignore_base:
            match_link = get_match_link(x, robot_id)
            if match_link is not None and robot_model.is_base_link(match_link):
                return False

        if ignore_names is not None:
            should_ignore = any(
                coll_name_matches(x, ignore_name)
                for ignore_name in ignore_names
            )
            if should_ignore:
                return False
        return True

    # Filter out any collisions with the ignore objects
    colls = list(filter(should_keep, colls))
    robot_coll_ids = []

    # Check for robot collision
    robot_obj_colls = 0
    robot_scene_colls = 0
    robot_scene_matches = [c for c in colls if coll_name_matches(c, robot_id)]
    for match in robot_scene_matches:
        reg_obj_coll = any(
            [coll_name_matches(match, obj_id) for obj_id in added_objs]
        )
        if reg_obj_coll:
            robot_obj_colls += 1
        else:
            robot_scene_colls += 1

        if match.object_id_a == robot_id:
            robot_coll_ids.append(match.object_id_b)
        else:
            robot_coll_ids.append(match.object_id_a)

    # Checking for holding object collision
    obj_scene_colls = 0
    if count_obj_colls and snapped_obj_id is not None:
        matches = [c for c in colls if coll_name_matches(c, snapped_obj_id)]
        for match in matches:
            if coll_name_matches(match, robot_id):
                continue
            obj_scene_colls += 1

    if get_extra_coll_data:
        coll_details = CollisionDetails(
            obj_scene_colls=min(obj_scene_colls, 1),
            robot_obj_colls=min(robot_obj_colls, 1),
            robot_scene_colls=min(robot_scene_colls, 1),
            robot_coll_ids=robot_coll_ids,
            all_colls=[(x.object_id_a, x.object_id_b) for x in colls],
        )
    else:
        coll_details = CollisionDetails(
            obj_scene_colls=min(obj_scene_colls, 1),
            robot_obj_colls=min(robot_obj_colls, 1),
            robot_scene_colls=min(robot_scene_colls, 1),
        )
    return coll_details.total_collisions > 0, coll_details


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
    obj = sim.get_rigid_object_manager().get_object_by_id(obj_id)
    if obj is None:
        return None
    obj_node = obj.root_scene_node
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


CACHE_PATH = "./data/cache"


class CacheHelper:
    def __init__(
        self, cache_name, lookup_val, def_val=None, verbose=False, rel_dir=""
    ):
        self.use_cache_path = osp.join(CACHE_PATH, rel_dir)
        os.makedirs(self.use_cache_path, exist_ok=True)
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


try:
    import pybullet as p
except ImportError:
    p = None


def is_pb_installed():
    return p is not None


class IkHelper:
    def __init__(self, only_arm_urdf, arm_start):
        self._arm_start = arm_start
        self._arm_len = 7
        self.pc_id = p.connect(p.DIRECT)

        self.robo_id = p.loadURDF(
            only_arm_urdf,
            basePosition=[0, 0, 0],
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.pc_id,
        )

        p.setGravity(0, 0, -9.81, physicsClientId=self.pc_id)
        JOINT_DAMPING = 0.5
        self.pb_link_idx = 7

        for link_idx in range(15):
            p.changeDynamics(
                self.robo_id,
                link_idx,
                linearDamping=0.0,
                angularDamping=0.0,
                jointDamping=JOINT_DAMPING,
                physicsClientId=self.pc_id,
            )
            p.changeDynamics(
                self.robo_id,
                link_idx,
                maxJointVelocity=200,
                physicsClientId=self.pc_id,
            )

    def set_arm_state(self, joint_pos, joint_vel=None):
        if joint_vel is None:
            joint_vel = np.zeros((len(joint_pos),))
        for i in range(7):
            p.resetJointState(
                self.robo_id,
                i,
                joint_pos[i],
                joint_vel[i],
                physicsClientId=self.pc_id,
            )

    def calc_fk(self, js):
        self.set_arm_state(js, np.zeros(js.shape))
        ls = p.getLinkState(
            self.robo_id,
            self.pb_link_idx,
            computeForwardKinematics=1,
            physicsClientId=self.pc_id,
        )
        world_ee = ls[4]
        return world_ee

    def get_joint_limits(self):
        lower = []
        upper = []
        for joint_i in range(self._arm_len):
            ret = p.getJointInfo(
                self.robo_id, joint_i, physicsClientId=self.pc_id
            )
            lower.append(ret[8])
            if ret[9] == -1:
                upper.append(2 * np.pi)
            else:
                upper.append(ret[9])
        return np.array(lower), np.array(upper)

    def calc_ik(self, targ_ee: np.ndarray):
        """
        :param targ_ee: 3D target position in the ROBOT BASE coordinate frame
        """
        js = p.calculateInverseKinematics(
            self.robo_id, self.pb_link_idx, targ_ee, physicsClientId=self.pc_id
        )
        return js[: self._arm_len]
