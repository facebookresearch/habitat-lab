#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import os.path as osp
import pickle
import time
from functools import wraps
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import attr
import magnum as mn
import numpy as np
import quaternion

import habitat_sim
from habitat.articulated_agents.mobile_manipulator import MobileManipulator
from habitat.articulated_agents.robots.spot_robot import SpotRobot
from habitat.articulated_agents.robots.stretch_robot import StretchRobot
from habitat.core.logging import HabitatLogger
from habitat.datasets.rearrange.navmesh_utils import snap_point_is_occluded
from habitat.tasks.utils import get_angle
from habitat_sim.physics import MotionType

if TYPE_CHECKING:
    # avoids circular import while allowing type hints
    from habitat.tasks.rearrange.rearrange_sim import RearrangeSim

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


def general_sim_collision(
    sim: habitat_sim.Simulator, agent_embodiement: MobileManipulator
) -> Tuple[bool, CollisionDetails]:
    """
    Proxy for "rearrange_collision()" which does not require a RearrangeSim.
    Used for testing functions which require a collision testing routine.

    :return: boolean flag denoting collisions and a details struct (not complete)
    """
    colls = sim.get_physics_contact_points()

    agent_embodiement_object_id = agent_embodiement.sim_obj.object_id

    robot_scene_colls = 0
    for col in colls:
        if coll_name_matches(col, agent_embodiement_object_id):
            robot_scene_colls += 1

    return (robot_scene_colls > 0), CollisionDetails(
        robot_scene_colls=robot_scene_colls
    )


def rearrange_collision(
    sim: "RearrangeSim",
    count_obj_colls: bool,
    ignore_names: Optional[List[str]] = None,
    ignore_base: bool = True,
    get_extra_coll_data: bool = False,
    agent_idx: Optional[int] = None,
) -> Tuple[bool, CollisionDetails]:
    """Defines what counts as a collision for the Rearrange environment execution"""
    agent_model = sim.get_agent_data(agent_idx).articulated_agent
    grasp_mgr = sim.get_agent_data(agent_idx).grasp_mgr
    colls = sim.get_physics_contact_points()
    agent_id = agent_model.get_robot_sim_id()
    added_objs = sim.scene_obj_ids
    snapped_obj_id = grasp_mgr.snap_idx

    def should_keep(x):
        if ignore_base:
            match_link = get_match_link(x, agent_id)
            if match_link is not None and agent_model.is_base_link(match_link):
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
    robot_scene_matches = [c for c in colls if coll_name_matches(c, agent_id)]
    for match in robot_scene_matches:
        reg_obj_coll = any(
            coll_name_matches(match, obj_id) for obj_id in added_objs
        )
        if reg_obj_coll:
            robot_obj_colls += 1
        else:
            robot_scene_colls += 1

        if match.object_id_a == agent_id:
            robot_coll_ids.append(match.object_id_b)
        else:
            robot_coll_ids.append(match.object_id_a)

    # Checking for holding object collision
    obj_scene_colls = 0
    if count_obj_colls and snapped_obj_id is not None:
        matches = [c for c in colls if coll_name_matches(c, snapped_obj_id)]
        for match in matches:
            if coll_name_matches(match, agent_id):
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


def get_rigid_aabb(
    obj_id: int, sim: habitat_sim.Simulator, transformed: bool = False
) -> Optional[mn.Range3D]:
    """
    Get the AABB for a RigidObject. Returns None if object is not found.

    :param obj_id: The unique id of the object instance.
    :param sim: The Simulator instance owning the object instance.
    :param transformed: If True, transform the AABB into global space.
    """
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


def get_ao_link_aabb(
    ao_obj_id: int,
    link_id: int,
    sim: habitat_sim.Simulator,
    transformed: bool = False,
) -> Optional[mn.Range3D]:
    """
    Get the AABB for a link of an ArticulatedObject. Returns None if object or link are not found.

    :param ao_obj_id: The unique id of the ArticulatedObject instance.
    :param link_id: The index of the link within the ArticulatedObject instance. -1 for base link. Note this is not unique object_id of the link.
    :param sim: The Simulator instance owning the object instance.
    :param transformed: If True, transform the AABB into global space.
    """

    ao = sim.get_articulated_object_manager().get_object_by_id(ao_obj_id)
    if ao is None:
        return None
    assert link_id < ao.num_links, "Link index out of range."
    link_node = ao.get_link_scene_node(link_id)
    link_bb = link_node.cumulative_bb
    if transformed:
        link_bb = habitat_sim.geo.get_transformed_bb(
            link_bb, link_node.absolute_transformation()
        )
    return link_bb


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


class UsesArticulatedAgentInterface:
    """
    For sensors or actions that are agent specific. Used to split actions and
    sensors between multiple agents.
    """

    def __init__(self, *args, **kwargs):
        # This init call is necessary for using this class with `Measure`.
        super().__init__(*args, **kwargs)
        self.agent_id = None


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


def set_agent_base_via_obj_trans(position: np.ndarray, rotation: float, agent):
    """Set the agent's base position and rotation via object transformation"""
    position = position - agent.sim_obj.transformation.transform_vector(
        agent.params.base_offset
    )
    quat = mn.Quaternion.rotation(
        mn.Rad(rotation), mn.Vector3(0, 1, 0)
    ).to_matrix()
    target_trans = mn.Matrix4.from_(quat, position)
    agent.sim_obj.transformation = target_trans


def get_angle_to_pos(rel_pos: np.ndarray) -> float:
    """
    Get the 1D orientation angle (around Y axis) for an agent with X axis forward to face toward a relative 3D position.

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


def add_perf_timing_func(name: Optional[str] = None):
    """
    Function decorator for logging the speed of a method to the RearrangeSim.
    This must either be applied to methods from RearrangeSim or to methods from
    objects that contain `self._sim` so this decorator can access the
    underlying `RearrangeSim` instance to log the speed. This scopes the
    logging name so nested function calls will include the outer perf timing
    name separate by a ".".

    :param name: The name of the performance logging key. If unspecified, this
        defaults to "ModuleName[FuncName]"
    """

    def perf_time(f):
        if name is None:
            module_name = f.__module__.split(".")[-1]
            use_name = f"{module_name}[{f.__name__}]"
        else:
            use_name = name

        @wraps(f)
        def wrapper(self, *args, **kwargs):
            if hasattr(self, "add_perf_timing") and hasattr(
                self, "cur_runtime_perf_scope"
            ):
                sim = self
            else:
                sim = self._sim

            if not hasattr(sim, "add_perf_timing"):
                # Does not support logging.
                return f(self, *args, **kwargs)

            sim.cur_runtime_perf_scope.append(use_name)
            t_start = time.time()
            ret = f(self, *args, **kwargs)
            sim.add_perf_timing("", t_start)
            sim.cur_runtime_perf_scope.pop()
            return ret

        return wrapper

    return perf_time


def get_camera_transform(cur_articulated_agent) -> mn.Matrix4:
    """Get the camera transformation"""
    if isinstance(cur_articulated_agent, SpotRobot):
        cam_info = cur_articulated_agent.params.cameras[
            "articulated_agent_arm_depth"
        ]
    elif isinstance(cur_articulated_agent, StretchRobot):
        cam_info = cur_articulated_agent.params.cameras["head"]
    else:
        raise NotImplementedError("This robot does not have GazeGraspAction.")

    # Get the camera's attached link
    link_trans = cur_articulated_agent.sim_obj.get_link_scene_node(
        cam_info.attached_link_id
    ).transformation
    # Get the camera offset transformation
    offset_trans = mn.Matrix4.translation(cam_info.cam_offset_pos)
    cam_trans = link_trans @ offset_trans @ cam_info.relative_transform
    return cam_trans


def angle_between(
    v1: Tuple[mn.Vector3, np.ndarray, List],
    v2: Tuple[mn.Vector3, np.ndarray, List],
) -> float:
    """Angle (in radians) between two vectors"""
    cosine = np.clip(np.dot(v1, v2), -1.0, 1.0)
    object_angle = np.arccos(cosine)
    return object_angle


def get_camera_object_angle(
    cam_T: mn.Matrix4,
    obj_pos: Tuple[mn.Vector3, np.ndarray, List],
    center_cone_vector: Tuple[mn.Vector3, np.ndarray, List],
) -> float:
    """Calculates angle between camera line-of-sight and given global position"""
    # Get object location in camera frame
    cam_obj_pos = cam_T.inverted().transform_point(obj_pos).normalized()
    # Get angle between (normalized) location and the vector that the camera should
    # look at
    obj_angle = angle_between(cam_obj_pos, center_cone_vector)
    return obj_angle


def get_camera_lookat_relative_to_vertial_line(
    agent,
) -> float:
    """Get the camera looking angles to a vertical line to the ground"""
    # Get the camera transformation
    cam_T = get_camera_transform(agent)
    # Get the camera position
    camera_pos = cam_T.translation
    # Cast a ray from the camera location to the ground
    vertical_dir = mn.Vector3(camera_pos[0], 0, camera_pos[2])
    # A true vertical line to the ground
    local_vertical_dir = mn.Vector3([0.0, 1.0, 0.0])
    # Get angle between location and the vector
    angle = get_camera_object_angle(cam_T, vertical_dir, local_vertical_dir)
    return angle


def embodied_unoccluded_navmesh_snap(
    target_position: mn.Vector3,
    height: float,
    sim: Union["RearrangeSim", habitat_sim.Simulator],
    pathfinder: habitat_sim.nav.PathFinder = None,
    target_object_id: Optional[int] = None,
    island_id: int = -1,
    search_offset: float = 1.5,
    test_batch_size: int = 20,
    max_samples: int = 200,
    min_sample_dist: float = 0.5,
    embodiement_heuristic_offsets: Optional[List[Tuple[float, float]]] = None,
    agent_embodiement: Optional[MobileManipulator] = None,
    orientation_noise: float = 0,
    max_orientation_samples: int = 5,
    data_out: Dict[Any, Any] = None,
) -> Tuple[mn.Vector3, float, bool]:
    """
    Snap a point to the navmesh considering point visibilty via raycasting.

    :property target_position: The 3D target position to snap.
    :property height: The height of the agent above the navmesh. Assumes the navmesh snap point is on the ground. Should be the maximum relative distance from navmesh ground to which a visibility check should indicate non-occlusion. The first check starts from this height. (E.g. agent_eyes_y - agent_base_y)
    :property sim: The RearrangeSimulator or Simulator instance. This choice will dictate the collision detection routine.
    :property pathfinder: The PathFinder defining the NavMesh to use.
    :property target_object_id: An optional object_id which should be ignored in the occlusion check. For example, when pos is an object's COM, that object should not occlude the point.
    :property island_id: Optionally restrict the search to a single navmesh island. Default -1 is the full navmesh.
    :property search_offset: The additional radius to search for navmesh points around the target position. Added to the minimum distance from pos to navmesh.
    :property test_batch_size: The number of sample navmesh points to consider when testing for occlusion.
    :property max_samples: The maximum number of attempts to sample navmesh points for the test batch.
    :property min_sample_dist: The minimum allowed L2 distance between samples in the test batch.
    :property embodiement_heuristic_offsets: A set of 2D offsets describing navmesh cylinder center points forming a proxy for agent embodiement. Assumes x-forward, y to the side and 3D height fixed to navmesh. If provided, this proxy embodiement will be used for collision checking.
    :property agent_embodiement: The MobileManipulator to be used for collision checking if provided.
    :property orientation_noise: Standard deviation of the gaussian used to sample orientation noise. If 0, states always face the target point. Noise is applied delta to this "target facing" orientation.
    :property max_orientation_samples: The number of orientation noise samples to try for each candidate point.
    :property data_out: Optionally provide a dictionary which can be filled with arbitrary detail data for external debugging and visualization.

    NOTE: this function is based on sampling and does not guarantee the closest point.

    :return: A Tuple containing: 1) An approximation of the closest unoccluded snap point to pos or None if an unoccluded point could not be found, 2) the sampled orientation if found or None, 3) a boolean success flag.
    """

    assert height > 0
    assert search_offset > 0
    assert test_batch_size > 0
    assert max_samples > 0
    assert orientation_noise >= 0

    if pathfinder is None:
        pathfinder = sim.pathfinder

    assert pathfinder.is_loaded

    # first try the closest snap point
    snap_point = pathfinder.snap_point(target_position, island_id)

    # distance to closest snap point is the absolute minimum
    min_radius = (snap_point - target_position).length()
    # expand the search radius
    search_radius = min_radius + search_offset

    # gather a test batch
    test_batch: List[Tuple[mn.Vector3, float]] = []
    sample_count = 0
    while len(test_batch) < test_batch_size and sample_count < max_samples:
        sample = pathfinder.get_random_navigable_point_near(
            circle_center=target_position,
            radius=search_radius,
            island_index=island_id,
        )
        reject = False
        for batch_sample in test_batch:
            if np.linalg.norm(sample - batch_sample[0]) < min_sample_dist:
                reject = True
                break
        if not reject:
            test_batch.append(
                (sample, float(np.linalg.norm(sample - target_position)))
            )
        sample_count += 1

    # sort the test batch points by distance to the target
    test_batch.sort(key=lambda s: s[1])

    # find the closest unoccluded point in the test batch
    for batch_sample in test_batch:
        if not snap_point_is_occluded(
            target_position,
            batch_sample[0],
            height,
            sim,
            target_object_id=target_object_id,
        ):
            facing_target_angle = get_angle_to_pos(
                np.array(target_position - batch_sample[0])
            )

            if (
                embodiement_heuristic_offsets is None
                and agent_embodiement is None
            ):
                # No embodiement for collision detection, so return closest unocluded point
                return batch_sample[0], facing_target_angle, True

            # get orientation noise offset
            orientation_noise_samples = []
            if orientation_noise > 0 and max_orientation_samples > 0:
                orientation_noise_samples = [
                    np.random.normal(0.0, orientation_noise)
                    for _ in range(max_orientation_samples)
                ]
            # last one is always no-noise to check forward-facing
            orientation_noise_samples.append(0)

            for orientation_noise_sample in orientation_noise_samples:
                desired_angle = facing_target_angle + orientation_noise_sample
                if embodiement_heuristic_offsets is not None:
                    # local 2d point rotation
                    rotation_2d = mn.Matrix3.rotation(-mn.Rad(desired_angle))
                    transformed_offsets_2d = [
                        rotation_2d.transform_vector(xz)
                        for xz in embodiement_heuristic_offsets
                    ]

                    # translation to global 3D points at navmesh height
                    offsets_3d = [
                        np.array(
                            [
                                transformed_offset_2d[0],
                                0,
                                transformed_offset_2d[1],
                            ]
                        )
                        + batch_sample[0]
                        for transformed_offset_2d in transformed_offsets_2d
                    ]

                    if data_out is not None:
                        data_out["offsets_3d"] = offsets_3d

                    # check for offset navigability
                    is_collision = False
                    for offset_point in offsets_3d:
                        if not (
                            sim.pathfinder.is_navigable(offset_point)
                            and (
                                island_id == -1
                                or sim.pathfinder.get_island(offset_point)
                                == island_id
                            )
                        ):
                            is_collision = True
                            break

                    # if this sample is invalid, try the next
                    if is_collision:
                        continue

                if agent_embodiement is not None:
                    # contact testing with collision shapes
                    start_position = agent_embodiement.base_pos
                    start_rotation = agent_embodiement.base_rot

                    agent_embodiement.base_pos = batch_sample[0]
                    agent_embodiement.base_rot = desired_angle

                    details = None
                    # Make sure the robot is not colliding with anything in this state.
                    if sim.__class__.__name__ == "RearrangeSim":
                        _, details = rearrange_collision(
                            sim,
                            False,
                            ignore_base=False,
                        )
                    else:
                        _, details = general_sim_collision(
                            sim, agent_embodiement
                        )

                    # reset agent state
                    agent_embodiement.base_pos = start_position
                    agent_embodiement.base_rot = start_rotation

                    # Only care about collisions between the robot and scene.
                    is_feasible_state = details.robot_scene_colls == 0
                    if not is_feasible_state:
                        continue

                # if we made it here, all tests passed and we found a valid placement state
                return batch_sample[0], desired_angle, True

    # unable to find a valid navmesh point within constraints
    return None, None, False
