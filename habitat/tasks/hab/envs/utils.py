import attr
import magnum as mn
import numpy as np
import pybullet as p
import quaternion
from PIL import Image

import habitat_sim
from habitat_sim.nav import NavMeshSettings
from habitat_sim.physics import MotionType

MP_TEST_DIR = "mp_test3"


def make_border_red(img):
    border_color = [255, 0, 0]
    border_width = 10
    img[:, :border_width] = border_color
    img[:border_width, :] = border_color
    img[-border_width:, :] = border_color
    img[:, -border_width:] = border_color
    return img


def get_angle(x, y):
    """
    Gets the angle between two vectors in radians.
    """
    if np.linalg.norm(x) != 0:
        x_norm = x / np.linalg.norm(x)
    else:
        x_norm = x

    if np.linalg.norm(y) != 0:
        y_norm = y / np.linalg.norm(y)
    else:
        y_norm = y
    return np.arccos(np.clip(np.dot(x_norm, y_norm), -1, 1))


def make_render_only(obj_idx, sim):
    if hasattr(MotionType, "RENDER_ONLY"):
        sim.set_object_motion_type(MotionType.RENDER_ONLY, obj_idx)
    else:
        sim.set_object_motion_type(MotionType.KINEMATIC, obj_idx)
        sim.set_object_is_collidable(False, obj_idx)


def has_collision(name, colls):
    for coll0, coll1 in colls:
        if coll0["name"] == name or coll1["name"] == name:
            return True
    return False


def get_collision_matches(link, colls, search_key="link"):
    matches = []
    for coll0, coll1 in colls:
        if coll0[search_key] == link or coll1[search_key] == link:
            matches.append((coll0, coll1))
    return matches


def get_other_matches(link, colls):
    matches = get_collision_matches(link, colls)
    other_surfaces = [b if a["link"] == link else a for a, b in matches]
    return other_surfaces


def coll_name(coll, name):
    return coll_prop(coll, name, "name")


def coll_prop(coll, val, prop):
    return coll[0][prop] == val or coll[1][prop] == val


def coll_link(coll, link):
    return coll_prop(coll, link, "link")


def swap_axes(x):
    x[1], x[2] = x[2], x[1]
    return x


class IkHelper:
    def __init__(self, arm_start):
        self._arm_start = arm_start
        self._arm_len = 7

    def setup_sim(self):
        self.pc_id = p.connect(p.DIRECT)

        self.robo_id = p.loadURDF(
            "./orp/robots/opt_fetch/robots/fetch_onlyarm.urdf",
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

    def calc_ik(self, targ_ee):
        """
        targ_ee is in ROBOT COORDINATE FRAME NOT IN EE COORDINATE FRAME
        """
        js = p.calculateInverseKinematics(
            self.robo_id, self.pb_link_idx, targ_ee, physicsClientId=self.pc_id
        )
        return js[: self._arm_len]


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
    ignore_names=[],
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

    def should_ignore(x):
        for ignore_name in ignore_names:
            if coll_name(x, ignore_name):
                return True
        return False

    # Filter out any collisions with the ignore objects
    colls = [x for x in colls if not should_ignore(x)]

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
    if count_obj_colls:
        if snapped_obj_id is not None:
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
        path = obj_dat[0]
        fname = '/'.join(path.split('/')[-2:])
        if '.urdf' in fname:
            obj_dat[0] = osp.join("data/misc_data/URDF", fname)

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


def inter_any_bb(bb0, bbs):
    for bb in bbs:
        if mn.math.intersects(bb0, bb):
            return True
    return False


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

    cur_r = sim.pathfinder.island_radius(v)
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


def get_largest_island_point(sim, height_thresh=None, z_min=None):
    """
    Samples a point from the largest island on the navmesh
    - height_thresh: Maximum possible height
    - z_thresh: Minimum possible z value.
    """
    use_vs = np.array(sim.pathfinder.build_navmesh_vertices())

    if height_thresh is not None:
        use_vs = use_vs[use_vs[:, 1] < height_thresh]
    if z_min is not None:
        use_vs = use_vs[use_vs[:, 2] > z_min]
    nav_vs_r = np.array([sim.pathfinder.island_radius(nav_v) for nav_v in use_vs])
    largest_island = np.max(nav_vs_r)
    use_vs = use_vs[nav_vs_r == largest_island]
    sel_i = np.random.randint(len(use_vs))

    ret = use_vs[sel_i]
    return ret


import hashlib
import os
import os.path as osp
import pickle
import time

CACHE_PATH = "./data/cache"

class CacheHelper:
    def __init__(self, cache_name, lookup_val, def_val=None, verbose=False,
            rel_dir=''):
        self.use_cache_path = osp.join(CACHE_PATH, rel_dir)
        if not osp.exists(self.use_cache_path):
            os.makedirs(self.use_cache_path)
        sec_hash = hashlib.md5(str(lookup_val).encode('utf-8')).hexdigest()
        cache_id = f"{cache_name}_{sec_hash}.pickle"
        self.cache_id = osp.join(self.use_cache_path, cache_id)
        self.def_val = def_val
        self.verbose = verbose

    def exists(self):
        return osp.exists(self.cache_id)

    def load(self, load_depth=0):
        if self.exists():
            try:
                with open(self.cache_id, 'rb') as f:
                    if self.verbose:
                        print('Loading cache @', self.cache_id)
                    return pickle.load(f)
            except EOFError as e:
                if load_depth == 32:
                    raise e
                # try again soon
                print("Cache size is ", osp.getsize(self.cache_id), 'for ', self.cache_id)
                time.sleep(1.0 + np.random.uniform(0.0, 1.0))
                return self.load(load_depth+1)
            return self.def_val
        else:
            return self.def_val


    def save(self, val):
        with open(self.cache_id, 'wb') as f:
            if self.verbose:
                print('Saving cache @', self.cache_id)
            pickle.dump(val, f)


import gym


def reshape_obs_space(obs_space, new_shape):
    assert isinstance(obs_space, gym.spaces.Box)
    return gym.spaces.Box(
        shape=new_shape,
        high=obs_space.low.reshape(-1)[0],
        low=obs_space.high.reshape(-1)[0],
        dtype=obs_space.dtype,
    )
