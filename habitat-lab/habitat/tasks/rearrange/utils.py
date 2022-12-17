#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import os.path as osp
import pickle
import time
from typing import List, Optional, Tuple

import attr
import magnum as mn
import numpy as np
import quaternion

import habitat_sim
from habitat.core.logging import HabitatLogger
from habitat.tasks.utils import get_angle
from habitat_sim.physics import MotionType

rearrange_logger = HabitatLogger(
    name="rearrange_task",
    level=int(os.environ.get("HABITAT_REARRANGE_LOG", logging.ERROR)),
    format_str="[%(levelname)s,%(name)s] %(asctime)-15s %(filename)s:%(lineno)d %(message)s",
)


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
    agent_idx: Optional[int] = None,
):
    """Defines what counts as a collision for the Rearrange environment execution"""
    robot_model = sim.get_robot_data(agent_idx).robot
    grasp_mgr = sim.get_robot_data(agent_idx).grasp_mgr
    colls = sim.get_physics_contact_points()
    robot_id = robot_model.get_robot_sim_id()
    added_objs = sim.scene_obj_ids
    snapped_obj_id = grasp_mgr.snap_idx

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


def convert_legacy_cfg(obj_list):
    if len(obj_list) == 0:
        return obj_list

    def convert_fn(obj_dat):
        fname = "/".join(obj_dat[0].split("/")[-2:])
        if ".urdf" in fname:
            obj_dat[0] = osp.join("data/replica_cad/urdf", fname)
        else:
            obj_dat[0] = obj_dat[0].replace(
                "data/objects/", "data/objects/ycb/configs/"
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


class CacheHelper:
    def __init__(self, cache_file, def_val=None, verbose=False):
        self.cache_id = cache_file
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
                    rearrange_logger.info(f"Loading cache @{self.cache_id}")
                return pickle.load(f)
        except EOFError as e:
            if load_depth == 32:
                raise e
            # try again soon
            rearrange_logger.warning(
                f"Cache size is {osp.getsize(self.cache_id)} for {self.cache_id}"
            )
            time.sleep(1.0 + np.random.uniform(0.0, 1.0))
            return self.load(load_depth + 1)

    def save(self, val):
        with open(self.cache_id, "wb") as f:
            if self.verbose:
                rearrange_logger.info(f"Saving cache @ {self.cache_id}")
            pickle.dump(val, f)


def batch_transform_point(
    points: np.ndarray, transform_matrix: mn.Matrix4, dtype=np.float32
) -> np.ndarray:
    transformed_points = []
    for point in points:
        transformed_points.append(transform_matrix.transform_point(point))
    return np.array(transformed_points, dtype=dtype)


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
        :param targ_ee: 3D target position in the robot BASE coordinate frame
        """
        js = p.calculateInverseKinematics(
            self.robo_id,
            self.pb_link_idx,
            targ_ee,
            physicsClientId=self.pc_id,
        )
        return js[: self._arm_len]


class UsesRobotInterface:
    """
    For sensors or actions that are robot specific. Used to split actions and
    sensors between multiple robots.
    """

    def __init__(self, *args, **kwargs):
        # This init call is necessary for using this class with `Measure`.
        super().__init__(*args, **kwargs)
        self.robot_id = None


def write_gfx_replay(gfx_keyframe_str, task_config, ep_id):
    """
    Writes the all replay frames to a file for later replay. Filename is of the
    form 'episodeX.replay.json' where `X` is the episode ID.
    """

    os.makedirs(task_config.gfx_replay_dir, exist_ok=True)
    # A gfx-replay list of keyframes for the episode. This is a JSON string that
    # should be saved to a file; the file can be read by visualization tools
    # (e.g. import into Blender for screenshots and videos).
    filepath = osp.join(
        task_config.gfx_replay_dir, f"episode{ep_id}.replay.json"
    )
    with open(filepath, "w") as text_file:
        text_file.write(gfx_keyframe_str)


def get_robot_spawns(
    target_position: np.ndarray,
    rotation_perturbation_noise: float,
    distance_threshold: int,
    sim,
    num_spawn_attempts: int,
    physics_stability_steps: int,
):
    """
    Attempts to place the robot near the target position, facing towards it

    :param target_position: The position of the target.
    :param rotation_perturbation_noise: The amount of noise to add to the robot's rotation.
    :param distance_threshold: The maximum distance from the target.
    :param sim: The simulator instance.
    :param num_spawn_attempts: The number of sample attempts for the distance threshold.
    :param physics_stability_steps: The number of steps to perform for physics stability check.

    :return: The robot's start position, rotation, and whether the placement was successful.
    """

    state = sim.capture_state()

    # Try to place the robot.
    for _ in range(num_spawn_attempts):
        sim.set_state(state)
        start_position = sim.pathfinder.get_random_navigable_point_near(
            target_position, distance_threshold
        )

        relative_target = target_position - start_position

        angle_to_object = get_angle_to_pos(relative_target)

        target_distance = np.linalg.norm(
            (start_position - target_position)[[0, 2]]
        )

        is_navigable = sim.pathfinder.is_navigable(start_position)

        # Face the robot towards the object.
        rotation_noise = np.random.normal(0.0, rotation_perturbation_noise)
        start_rotation = angle_to_object + rotation_noise

        if target_distance > distance_threshold or not is_navigable:
            continue

        sim.robot.base_pos = start_position
        sim.robot.base_rot = start_rotation

        # Make sure the robot is not colliding with anything in this
        # position.
        for _ in range(physics_stability_steps):
            sim.perform_discrete_collision_detection()
            _, details = rearrange_collision(
                sim,
                False,
                ignore_base=False,
            )

            # Only care about collisions between the robot and scene.
            did_collide = details.robot_scene_colls != 0

            if did_collide:
                break

        if not did_collide:
            sim.set_state(state)
            return start_position, start_rotation, False

    sim.set_state(state)
    return start_position, start_rotation, True


def get_angle_to_pos(rel_pos: np.ndarray) -> float:
    """
    :param rel_pos: Relative 3D positive from the robot to the target like: `target_pos - robot_pos`.
    :returns: Angle in radians.
    """

    forward = np.array([1.0, 0, 0])
    rel_pos = np.array(rel_pos)
    forward = forward[[0, 2]]
    rel_pos = rel_pos[[0, 2]]

    heading_angle = get_angle(forward, rel_pos)
    c = np.cross(forward, rel_pos) < 0
    if not c:
        heading_angle = -1.0 * heading_angle
    return heading_angle
