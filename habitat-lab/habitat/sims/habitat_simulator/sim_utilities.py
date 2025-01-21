#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module provides a diverse set of functional utilities for common operations involving the Simulator and ManagedObjects including: object getters from id and handle, geometric utilities, prepositional logic, region queries, articulated object interactions, and more."""
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import magnum as mn
import numpy as np
from scipy import spatial

import habitat_sim
from habitat.sims.habitat_simulator.debug_visualizer import DebugVisualizer

if TYPE_CHECKING:
    from habitat.datasets.rearrange.samplers.receptacle import Receptacle


def object_shortname_from_handle(object_handle: str) -> str:
    """
    Splits any path directory and instance increment from the handle.

    :param object_handle: The raw object template or instance handle.
    :return: the shortened name string.
    """

    return object_handle.split("/")[-1].split(".")[0].split("_:")[0]


def get_bb_corners(range3d: mn.Range3D) -> List[mn.Vector3]:
    """
    Get the corner points for an Axis-aligned bounding box (AABB).

    :param range3d: The bounding box for which to get the corners.
    :return: a list of AABB (Range3D) corners in object local space.
    """
    return [
        range3d.back_bottom_left,
        range3d.back_bottom_right,
        range3d.back_top_right,
        range3d.back_top_left,
        range3d.front_top_left,
        range3d.front_top_right,
        range3d.front_bottom_right,
        range3d.front_bottom_left,
    ]


def get_bb_for_object_id(
    sim: habitat_sim.Simulator,
    obj_id: int,
    ao_link_map: Dict[int, int] = None,
) -> Tuple[mn.Range3D, mn.Matrix4]:
    """
    Wrapper to get a bb and global transform directly from an object id.
    Handles RigidObject and ArticulatedLink ids.

    :param sim: The Simulator instance.
    :param obj_id: The integer id of the object or link.
    :param ao_link_map: A pre-computed map from link object ids to their parent ArticulatedObject's object id.
    :return: tuple (local_aabb, global_transform)
    """

    # stage bounding box
    if obj_id == habitat_sim.stage_id:
        return (
            sim.get_active_scene_graph().get_root_node().cumulative_bb,
            mn.Matrix4.identity_init(),
        )

    obj = get_obj_from_id(sim, obj_id, ao_link_map)

    if obj is None:
        raise AssertionError(
            f"object id {obj_id} is not found, this is unexpected. Invalid/stale object id?"
        )

    # ManagedObject
    if obj.object_id == obj_id:
        return (obj.aabb, obj.transformation)

    # this is a link
    link_node = obj.get_link_scene_node(obj.link_object_ids[obj_id])
    link_transform = link_node.absolute_transformation()
    return (link_node.cumulative_bb, link_transform)


def get_obj_size_along(
    sim: habitat_sim.Simulator,
    object_id: int,
    global_vec: mn.Vector3,
    ao_link_map: Dict[int, int] = None,
) -> Tuple[float, mn.Vector3]:
    """
    Uses object bounding box ellipsoid scale as a heuristic to estimate object size in a particular global direction.

    :param sim: The Simulator instance.
    :param object_id: The integer id of the object or link.
    :param global_vec: Vector in global space indicating the direction to approximate object size.
    :param ao_link_map: A pre-computed map from link object ids to their parent ArticulatedObject's object id.
    :return: distance along the specified direction and global center of bounding box from which distance was estimated.
    """

    obj_bb, transform = get_bb_for_object_id(sim, object_id, ao_link_map)
    center = transform.transform_point(obj_bb.center())
    local_scale = mn.Matrix4.scaling(obj_bb.size() / 2.0)
    local_vec = transform.inverted().transform_vector(global_vec).normalized()
    local_vec_size = local_scale.transform_vector(local_vec).length()
    return local_vec_size, center


def size_regularized_bb_distance(
    bb_a: mn.Range3D,
    bb_b: mn.Range3D,
    transform_a: mn.Matrix4 = None,
    transform_b: mn.Matrix4 = None,
    flatten_axis: int = None,
) -> float:
    """
    Get the heuristic surface-to-surface distance between two bounding boxes (regularized by their individual heuristic sizes).
    Estimate the distance from center to boundary along the line between bb centers. These sizes are then subtracted from the center-to-center distance as a heuristic for surface-to-surface distance.

    :param bb_a: local bounding box of one object
    :param bb_b: local bounding box of another object
    :param transform_a: local to global transform for the first object. Default is identity.
    :param transform_b: local to global transform for the second object. Default is identity.
    :param flatten_axis: Optionally flatten one axis of the displacement vector. This effectively projects the displacement. For example, index "1" would result in horizontal (xz) distance.
    :return: heuristic surface-to-surface distance.
    """

    # check for a valid value
    assert flatten_axis in [None, 0, 1, 2]

    if transform_a is None:
        transform_a = mn.Matrix4.identity_init()
    if transform_b is None:
        transform_b = mn.Matrix4.identity_init()

    a_center = transform_a.transform_point(bb_a.center())
    b_center = transform_b.transform_point(bb_b.center())

    disp = a_center - b_center
    # optionally project the displacement vector by flattening it
    if flatten_axis is not None:
        disp[flatten_axis] = 0
    dist = disp.length()
    disp_dir = disp / dist

    local_scale_a = mn.Matrix4.scaling(bb_a.size() / 2.0)
    local_vec_a = transform_a.inverted().transform_vector(disp_dir)
    local_vec_size_a = local_scale_a.transform_vector(local_vec_a).length()

    local_scale_b = mn.Matrix4.scaling(bb_b.size() / 2.0)
    local_vec_b = transform_b.inverted().transform_vector(disp_dir)
    local_vec_size_b = local_scale_b.transform_vector(local_vec_b).length()

    # if object bounding boxes are significantly overlapping then distance may be negative, clamp to 0
    return max(0, dist - local_vec_size_a - local_vec_size_b)


def size_regularized_object_distance(
    sim: habitat_sim.Simulator,
    object_id_a: int,
    object_id_b: int,
    ao_link_map: Dict[int, int] = None,
) -> float:
    """
    Get the heuristic surface-to-surface distance between two objects (regularized by their individual heuristic sizes).
    Uses each object's bounding box to estimate the distance from center to boundary along the line between object centers. These object sizes are then subtracted from the center-to-center distance as a heuristic for surface-to-surface distance.

    :param sim: The Simulator instance.
    :param object_id_a: integer id of the first object
    :param object_id_b: integer id of the second object
    :param ao_link_map: A pre-computed map from link object ids to their parent ArticulatedObject's object id.
    :return: The heuristic surface-2-surface distance between the objects.
    """

    # distance to self
    if object_id_a == object_id_b:
        return 0

    assert (
        object_id_a != habitat_sim.stage_id
        and object_id_b != habitat_sim.stage_id
    ), "Cannot compute distance between the scene and its contents."

    obja_bb, transform_a = get_bb_for_object_id(sim, object_id_a, ao_link_map)
    objb_bb, transform_b = get_bb_for_object_id(sim, object_id_b, ao_link_map)

    return size_regularized_bb_distance(
        obja_bb, objb_bb, transform_a, transform_b
    )


def bb_ray_prescreen(
    sim: habitat_sim.Simulator,
    obj: habitat_sim.physics.ManagedRigidObject,
    support_obj_ids: Optional[List[int]] = None,
    ignore_obj_ids: Optional[List[int]] = None,
    check_all_corners: bool = False,
) -> Dict[str, Any]:
    """
    Pre-screen a potential placement by casting rays in the gravity direction from the object center of mass (and optionally each corner of its bounding box) checking for interferring objects below.

    :param sim: The Simulator instance.
    :param obj: The RigidObject instance.
    :param support_obj_ids: A list of object ids designated as valid support surfaces for object placement. Contact with other objects is a criteria for placement rejection.
    :param ignore_obj_ids: A list of object ids which should be ignored in contact checks and raycasts. For example, the body of the agent placing an object.
    :param check_all_corners: Optionally cast rays from all bounding box corners instead of only casting a ray from the center of mass.
    :return: a dict of raycast metadata: "base_rel_height","surface_snap_point", "raycast_results"
    """

    if support_obj_ids is None:
        # set default support surface to stage/ground mesh
        # STAGE ID IS habitat_sim.stage_id
        support_obj_ids = [habitat_sim.stage_id]
    lowest_key_point: mn.Vector3 = None
    lowest_key_point_height = None
    highest_support_impact: Optional[mn.Vector3] = None
    highest_support_impact_height = None
    highest_support_impact_id = None
    raycast_results = []
    gravity_dir = sim.get_gravity().normalized()
    object_local_to_global = obj.transformation
    col_aabb = obj.collision_shape_aabb
    render_aabb = obj.aabb
    # use the full collision and render aabb for best snapping accuracy
    joined_aabb = mn.Range3D(
        mn.math.min(col_aabb.min, render_aabb.min),
        mn.math.max(col_aabb.max, render_aabb.max),
    )
    bb_corners = get_bb_corners(joined_aabb)

    key_points = [mn.Vector3(0)] + bb_corners  # [COM, c0, c1 ...]
    support_impacts: Dict[int, mn.Vector3] = {}  # indexed by keypoints
    for ix, key_point in enumerate(key_points):
        world_point = object_local_to_global.transform_point(key_point)
        # NOTE: instead of explicit Y coordinate, we project onto any gravity vector
        world_point_height = world_point.projected_onto_normalized(
            -gravity_dir
        ).length()
        if (
            lowest_key_point is None
            or lowest_key_point_height > world_point_height
        ):
            lowest_key_point = world_point
            lowest_key_point_height = world_point_height
        # cast a ray in gravity direction
        if ix == 0 or check_all_corners:
            ray = habitat_sim.geo.Ray(world_point, gravity_dir)
            raycast_results.append(sim.cast_ray(ray))
            # classify any obstructions before hitting the support surface
            for hit in raycast_results[-1].hits:
                if (
                    hit.object_id == obj.object_id
                    or ignore_obj_ids is not None
                    and hit.object_id in ignore_obj_ids
                ):
                    continue
                elif hit.object_id in support_obj_ids:
                    hit_point = hit.point
                    support_impacts[ix] = hit_point
                    support_impact_height = mn.math.dot(
                        hit_point, -gravity_dir
                    )

                    if (
                        highest_support_impact is None
                        or highest_support_impact_height
                        < support_impact_height
                    ):
                        highest_support_impact = hit_point
                        highest_support_impact_height = support_impact_height
                        highest_support_impact_id = hit.object_id

                # terminates at the first non-self ray hit
                break

    # compute the relative base height of the object from its lowest bb corner and COM
    base_rel_height = (
        lowest_key_point_height
        - obj.translation.projected_onto_normalized(-gravity_dir).length()
    )

    # account for the affects of stage mesh margin
    # Warning: Bullet raycast on stage triangle mesh does NOT consider the margin, so explicitly consider this here.
    margin_offset = 0
    if highest_support_impact_id is None:
        pass
    elif highest_support_impact_id == habitat_sim.stage_id:
        margin_offset = sim.get_stage_initialization_template().margin

    surface_snap_point = (
        None
        if 0 not in support_impacts
        else highest_support_impact
        + gravity_dir * (base_rel_height - margin_offset)
    )

    # return list of relative base height, object position for surface snapped point, and ray results details
    return {
        "base_rel_height": base_rel_height,
        "surface_snap_point": surface_snap_point,
        "raycast_results": raycast_results,
    }


def snap_down(
    sim: habitat_sim.Simulator,
    obj: habitat_sim.physics.ManagedRigidObject,
    support_obj_ids: Optional[List[int]] = None,
    ignore_obj_ids: Optional[List[int]] = None,
    dbv: Optional[DebugVisualizer] = None,
    max_collision_depth: float = 0.01,
) -> bool:
    """
    Attempt to project an object in the gravity direction onto the surface below it.

    :param sim: The Simulator instance.
    :param obj: The RigidObject instance.
    :param support_obj_ids: A list of object ids designated as valid support surfaces for object placement. Contact with other objects is a criteria for placement rejection. If none provided, default support surface is the stage/ground mesh (0).
    :param ignore_obj_ids: A list of object ids which should be ignored in contact checks and raycasts. For example, the body of the agent placing an object.
    :param dbv: Optionally provide a DebugVisualizer (dbv) to render debug images of each object's computed snap position before collision culling.
    :param max_collision_depth: The maximum contact penetration depth between the object and the support surface. Higher values are easier to sample, but result in less dynamically stabile states.
    :return: boolean placement success.

    Reject invalid placements by checking for penetration with other existing objects.
    If placement is successful, the object state is updated to the snapped location.
    If placement is rejected, object position is not modified and False is returned.

    To use this utility, generate an initial placement for any object above any of the designated support surfaces and call this function to attempt to snap it onto the nearest surface in the gravity direction.
    """

    aom = sim.get_articulated_object_manager()

    cached_position = obj.translation

    if support_obj_ids is None:
        # set default support surface to stage/ground mesh
        support_obj_ids = [habitat_sim.stage_id]

    if ignore_obj_ids is None:
        # default empty to avoid extra none checks in-loop later
        ignore_obj_ids = []

    bb_ray_prescreen_results = bb_ray_prescreen(
        sim, obj, support_obj_ids, ignore_obj_ids, check_all_corners=False
    )

    if bb_ray_prescreen_results["surface_snap_point"] is None:
        # no support under this object, return failure
        return False

    # finish up
    if bb_ray_prescreen_results["surface_snap_point"] is not None:
        # accept the final location if a valid location exists
        obj.translation = bb_ray_prescreen_results["surface_snap_point"]
        if dbv is not None:
            dbv.debug_obs.append(dbv.get_observation(obj.translation))
        sim.perform_discrete_collision_detection()
        cps = sim.get_physics_contact_points()
        for cp in cps:
            if (
                # the object is involved in the contact
                cp.object_id_a == obj.object_id
                or cp.object_id_b == obj.object_id
            ):
                cp_obj_id_a = cp.object_id_a
                cp_obj_id_b = cp.object_id_b
                if cp.link_id_a > 0:
                    # object_a is an AO and we need to get the link object id
                    ao_a = aom.get_object_by_id(cp.object_id_a)
                    links_to_obj_ids = {
                        v: k for k, v in ao_a.link_object_ids.items()
                    }
                    cp_obj_id_a = links_to_obj_ids[cp.link_id_a]
                if cp.link_id_b > 0:
                    # object_b is an AO and we need to get the link object id
                    ao_b = aom.get_object_by_id(cp.object_id_b)
                    links_to_obj_ids = {
                        v: k for k, v in ao_b.link_object_ids.items()
                    }
                    cp_obj_id_b = links_to_obj_ids[cp.link_id_b]

                if not (
                    # the contact does not involve ignored objects
                    cp_obj_id_a in ignore_obj_ids
                    or cp_obj_id_b in ignore_obj_ids
                ) and (
                    not (
                        # contact is not with a support object
                        cp_obj_id_a in support_obj_ids
                        or cp_obj_id_b in support_obj_ids
                    )
                    or (
                        # contact exceeds maximum depth
                        # NOTE: contact depth is negative distance
                        cp.contact_distance
                        < (-1 * max_collision_depth)
                    )
                ):
                    obj.translation = cached_position
                    return False
                    # print(f" Failure: contact in final position w/ distance = {cp.contact_distance}.")
                    # print(f" Failure: contact in final position with non support object {cp.object_id_a} or {cp.object_id_b}.")

        return True
    else:
        # no valid position found, reset and return failure
        obj.translation = cached_position
        return False


def get_all_object_ids(sim: habitat_sim.Simulator) -> Dict[int, str]:
    """
    Generate a dict mapping all active object ids to a descriptive string containing the object instance handle and, for ArticulatedLinks, the link name.

    :param sim: The Simulator instance.
    :return: a dict mapping object ids to a descriptive string.
    """

    rom = sim.get_rigid_object_manager()
    aom = sim.get_articulated_object_manager()

    object_id_map = {}

    for _object_handle, rigid_object in rom.get_objects_by_handle_substring(
        ""
    ).items():
        object_id_map[rigid_object.object_id] = rigid_object.handle

    for _object_handle, ao in aom.get_objects_by_handle_substring("").items():
        object_id_map[ao.object_id] = ao.handle
        for object_id, link_ix in ao.link_object_ids.items():
            object_id_map[object_id] = (
                ao.handle + " -- " + ao.get_link_name(link_ix)
            )

    return object_id_map


def get_all_objects(
    sim: habitat_sim.Simulator,
) -> List[
    Union[
        habitat_sim.physics.ManagedRigidObject,
        habitat_sim.physics.ManagedArticulatedObject,
    ]
]:
    """
    Get a list of all ManagedRigidObjects and ManagedArticulatedObjects in the scene.

    :param sim: The Simulator instance.
    :return: a list of ManagedObject wrapper instances containing all objects currently instantiated in the scene.
    """

    managers = [
        sim.get_rigid_object_manager(),
        sim.get_articulated_object_manager(),
    ]
    all_objects = []
    for mngr in managers:
        all_objects.extend(mngr.get_objects_by_handle_substring().values())
    return all_objects


def get_ao_link_id_map(sim: habitat_sim.Simulator) -> Dict[int, int]:
    """
    Construct a dict mapping ArticulatedLink object_id to parent ArticulatedObject object_id.
    NOTE: also maps ao's root object id to itself for ease of use.

    :param sim: The Simulator instance.
    :return: dict mapping ArticulatedLink object ids to parent object ids.
    """

    aom = sim.get_articulated_object_manager()
    ao_link_map: Dict[int, int] = {}
    for ao in aom.get_objects_by_handle_substring().values():
        # add the ao itself for ease of use
        ao_link_map[ao.object_id] = ao.object_id
        # add the links
        for link_id in ao.link_object_ids:
            ao_link_map[link_id] = ao.object_id

    return ao_link_map


def get_ao_default_link(
    ao: habitat_sim.physics.ManagedArticulatedObject,
    compute_if_not_found: bool = False,
) -> Optional[int]:
    """
    Get the "default" link index for a ManagedArticulatedObject.
    The "default" link is the one link which should be used if only one joint can be actuated. For example, the largest or most accessible drawer or door.

    :param ao: The ManagedArticulatedObject instance.
    :param compute_if_not_found: If true, try to compute the default link if it isn't found.
    :return: The default link index or None if not found. Cannot be base link (-1).

    The default link is determined by:

        - must be "prismatic" or "revolute" joint type
        - first look in the metadata Configuration for an annotated link.
        - (if compute_if_not_found) - if not annotated, it is programmatically computed from a heuristic.

    Default link heuristic: the link with the lowest Y value in the bounding box with appropriate joint type.
    """

    # first look in metadata
    default_link = ao.user_attributes.get("default_link")

    if default_link is None and compute_if_not_found:
        valid_joint_types = [
            habitat_sim.physics.JointType.Revolute,
            habitat_sim.physics.JointType.Prismatic,
        ]
        lowest_link = None
        lowest_y: int = None
        # compute the default link
        for link_id in ao.get_link_ids():
            if ao.get_link_joint_type(link_id) in valid_joint_types:
                # use minimum global keypoint Y value
                link_lowest_y = min(
                    get_articulated_link_global_keypoints(ao, link_id),
                    key=lambda x: x[1],
                )[1]
                if lowest_y is None or link_lowest_y < lowest_y:
                    lowest_y = link_lowest_y
                    lowest_link = link_id
        if lowest_link is not None:
            default_link = lowest_link
            # if found, set in metadata for next time
            ao.user_attributes.set("default_link", default_link)

    return default_link


def get_obj_from_id(
    sim: habitat_sim.Simulator,
    obj_id: int,
    ao_link_map: Optional[Dict[int, int]] = None,
) -> Union[
    habitat_sim.physics.ManagedRigidObject,
    habitat_sim.physics.ManagedArticulatedObject,
]:
    """
    Get a ManagedRigidObject or ManagedArticulatedObject from an object_id.

    :param sim: The Simulator instance.
    :param obj_id: object id for which ManagedObject is desired.
    :param ao_link_map: A pre-computed map from link object ids to their parent ArticulatedObject's object id.
    :return: a ManagedObject or None

    ArticulatedLink object_ids will return the ManagedArticulatedObject.
    If you want link id, use ManagedArticulatedObject.link_object_ids[obj_id].
    """

    rom = sim.get_rigid_object_manager()
    if rom.get_library_has_id(obj_id):
        return rom.get_object_by_id(obj_id)

    if ao_link_map is None:
        # Note: better to pre-compute this and pass it around
        ao_link_map = get_ao_link_id_map(sim)

    aom = sim.get_articulated_object_manager()
    if obj_id in ao_link_map:
        return aom.get_object_by_id(ao_link_map[obj_id])

    return None


def get_obj_from_handle(
    sim: habitat_sim.Simulator, obj_handle: str
) -> Union[
    habitat_sim.physics.ManagedRigidObject,
    habitat_sim.physics.ManagedArticulatedObject,
]:
    """
    Get a ManagedRigidObject or ManagedArticulatedObject from its instance handle.

    :param sim: The Simulator instance.
    :param obj_handle: object instance handle for which ManagedObject is desired.
    :return: a ManagedObject or None
    """

    rom = sim.get_rigid_object_manager()
    if rom.get_library_has_handle(obj_handle):
        return rom.get_object_by_handle(obj_handle)
    aom = sim.get_articulated_object_manager()
    if aom.get_library_has_handle(obj_handle):
        return aom.get_object_by_handle(obj_handle)

    return None


def get_obj_transform_from_id(
    sim: habitat_sim.Simulator,
    obj_id: int,
    ao_link_map: Optional[Dict[int, int]] = None,
) -> mn.Matrix4:
    """
    Retrieve the local to global transform of the object or link identified by the object_id.

    :param sim: The Simulator instance.
    :param obj_id: object id for which ManagedObject is desired.
    :param ao_link_map: A pre-computed map from link object ids to their parent ArticulatedObject's object id.
    :return: a Matrix4 local to global transform or None
    """

    parent_obj = get_obj_from_id(sim, obj_id, ao_link_map)
    if parent_obj is None:
        # invalid object id
        return None

    if parent_obj.object_id == obj_id:
        # this is a rigid or articulated object
        return parent_obj.transformation
    else:
        # this is a link
        return parent_obj.get_link_scene_node(
            parent_obj.link_object_ids[obj_id]
        ).transformation


def get_global_keypoints_from_bb(
    aabb: mn.Range3D, local_to_global: mn.Matrix4
) -> List[mn.Vector3]:
    """
    Get a list of bounding box keypoints in global space.
    0th point is the bounding box center, others are bounding box corners.

    :param aabb: The local bounding box.
    :param local_to_global: The local to global transformation matrix.
    :return: A set of global 3D keypoints for the bounding box.
    """
    local_keypoints = [aabb.center()]
    local_keypoints.extend(get_bb_corners(aabb))
    global_keypoints = [
        local_to_global.transform_point(key_point)
        for key_point in local_keypoints
    ]
    return global_keypoints


def get_articulated_link_global_keypoints(
    object_a: habitat_sim.physics.ManagedArticulatedObject, link_index: int
) -> List[mn.Vector3]:
    """
    Get global bb keypoints for an ArticulatedLink.

    :param object_a: The parent ManagedArticulatedObject for the link.
    :param link_index: The local index of the link within the parent ArticulatedObject. Not the object_id of the link.
    :return: A set of global 3D keypoints for the link.
    """
    link_node = object_a.get_link_scene_node(link_index)

    return get_global_keypoints_from_bb(
        link_node.cumulative_bb, link_node.absolute_transformation()
    )


def get_global_keypoints_from_object_id(
    sim: habitat_sim.Simulator,
    object_id: int,
    ao_link_map: Optional[Dict[int, int]] = None,
) -> List[mn.Vector3]:
    """
    Get a list of object keypoints in global space given an object id.
    0th point is the center of bb, others are bounding box corners.

    :param sim: The Simulator instance.
    :param object_id: The integer id for the object from which to extract keypoints.
    :param ao_link_map: A pre-computed map from link object ids to their parent ArticulatedObject's object id. If not provided, recomputed as necessary.
    :return: A set of global 3D keypoints for the object.
    """

    obj = get_obj_from_id(sim, object_id, ao_link_map)

    if obj.object_id != object_id:
        # this is an ArticulatedLink
        return get_articulated_link_global_keypoints(
            obj, obj.link_object_ids[object_id]
        )
    else:
        # ManagedObject
        return get_global_keypoints_from_bb(obj.aabb, obj.transformation)


def object_keypoint_cast(
    sim: habitat_sim.Simulator,
    object_a: habitat_sim.physics.ManagedRigidObject,
    direction: mn.Vector3 = None,
) -> List[habitat_sim.physics.RaycastResults]:
    """
    Computes object global keypoints, casts rays from each in the specified direction and returns the resulting RaycastResults.

    :param sim: The Simulator instance.
    :param object_a: The ManagedRigidObject from which to extract keypoints and raycast.
    :param direction: Optionally provide a unit length global direction vector for the raycast. If None, default to -Y.
    :return: A list of RaycastResults, one from each object keypoint.
    """

    if direction is None:
        # default to downward raycast
        direction = mn.Vector3(0, -1, 0)

    global_keypoints = get_global_keypoints_from_bb(
        object_a.aabb, object_a.transformation
    )
    return [
        sim.cast_ray(habitat_sim.geo.Ray(keypoint, direction))
        for keypoint in global_keypoints
    ]


# ============================================================
# Utilities for Querying Object Relationships
# ============================================================


def above(
    sim: habitat_sim.Simulator,
    object_a: Union[
        habitat_sim.physics.ManagedRigidObject,
        habitat_sim.physics.ManagedArticulatedObject,
    ],
) -> List[int]:
    """
    Get a list of all objects that a particular object_a is 'above'.
    Concretely, 'above' is defined as: a downward raycast of any object keypoint hits the object below.

    :param sim: The Simulator instance.
    :param object_a: The ManagedRigidObject for which to query the 'above' set.
    :return: a list of object ids.
    """

    # get object ids of all objects below this one
    above_object_ids = [
        hit.object_id
        for keypoint_raycast_result in object_keypoint_cast(sim, object_a)
        for hit in keypoint_raycast_result.hits
    ]
    above_object_ids = list(set(above_object_ids))

    # remove self from the list if present
    if object_a.object_id in above_object_ids:
        above_object_ids.remove(object_a.object_id)

    return above_object_ids


def within(
    sim: habitat_sim.Simulator,
    object_a: Union[
        habitat_sim.physics.ManagedRigidObject,
        habitat_sim.physics.ManagedArticulatedObject,
    ],
    max_distance: float = 1.0,
    keypoint_vote_threshold: int = 2,
    center_ensures_containment: bool = True,
) -> List[int]:
    """
    Get a list of all objects that a particular object_a is 'within'.
    Concretely, 'within' is defined as: a threshold number of opposing keypoint raycasts hit the same object.
    This function computes raycasts along all global axes from all keypoints and checks opposing rays for collision with the same object.

    :param sim: The Simulator instance.
    :param object_a: The ManagedRigidObject for which to query the 'within' set.
    :param max_distance: The maximum ray distance to check in each opposing direction (this is half the "wingspan" of the check). Makes the raycast more efficienct and realistically containing objects will have a limited size.
    :param keypoint_vote_threshold: The minimum number of keypoints which must indicate containment to qualify object_a as "within" another object.
    :param center_ensures_containment: If True, positive test of object_a's center keypoint alone qualifies object_a as "within" another object.
    :return: a list of object_id integers.
    """

    global_keypoints = get_global_keypoints_from_object_id(
        sim, object_a.object_id
    )

    # build axes vectors
    pos_axes = [mn.Vector3.x_axis(), mn.Vector3.y_axis(), mn.Vector3.z_axis()]
    neg_axes = [-1 * axis for axis in pos_axes]

    # raycast for each axis for each keypoint
    keypoint_intersect_set: List[List[int]] = [
        [] for _ in range(len(global_keypoints))
    ]
    for k_ix, keypoint in enumerate(global_keypoints):
        for a_ix in range(3):
            pos_ids = [
                hit.object_id
                for hit in sim.cast_ray(
                    habitat_sim.geo.Ray(keypoint, pos_axes[a_ix]),
                    max_distance=max_distance,
                ).hits
            ]
            neg_ids = [
                hit.object_id
                for hit in sim.cast_ray(
                    habitat_sim.geo.Ray(keypoint, neg_axes[a_ix]),
                    max_distance=max_distance,
                ).hits
            ]
            intersect_ids = [obj_id for obj_id in pos_ids if obj_id in neg_ids]
            keypoint_intersect_set[k_ix].extend(intersect_ids)
        keypoint_intersect_set[k_ix] = list(set(keypoint_intersect_set[k_ix]))

    containment_ids = []

    # used to toggle "center" keypoint as a voting or overriding check
    first_voting_keypoint = 0

    if center_ensures_containment:
        # initialize the list from keypoint 0 (center of bounding box) which guarantees containment
        containment_ids = list(keypoint_intersect_set[0])
        first_voting_keypoint = 1

    # "vote" for ids from keypoints
    id_votes: defaultdict[int, int] = defaultdict(lambda: 0)
    for k_ix in range(first_voting_keypoint, len(global_keypoints)):
        for obj_id in keypoint_intersect_set[k_ix]:
            id_votes[obj_id] += 1

    # count votes and de-duplicate
    containment_ids = containment_ids + [
        obj_id
        for obj_id in id_votes
        if id_votes[obj_id] > keypoint_vote_threshold
    ]
    containment_ids = list(set(containment_ids))

    # remove self from the list if present
    if object_a.object_id in containment_ids:
        containment_ids.remove(object_a.object_id)

    return containment_ids


def ontop(
    sim: habitat_sim.Simulator,
    object_a: Union[
        habitat_sim.physics.ManagedRigidObject,
        habitat_sim.physics.ManagedArticulatedObject,
        int,
    ],
    do_collision_detection: bool,
    vertical_normal_error_threshold: float = 0.75,
) -> List[int]:
    """
    Get a list of all object ids or objects that are "ontop" of a particular object_a.
    Concretely, 'ontop' is defined as: contact points between object_a and object_b have vertical normals "upward" relative to object_a.
    This function uses collision points to determine which objects are resting on or contacting the surface of object_a.

    :param sim: The Simulator instance.
    :param object_a: The ManagedRigidObject or object id for which to query the 'ontop' set.
    :param do_collision_detection: If True, a fresh discrete collision detection is run before the contact point query. Pass False to skip if a recent sim step or pre-process has run a collision detection pass on the current state.
    :param vertical_normal_error_threshold: The allowed error in normal alignment for a contact point to be considered "vertical" for this check. Functionally, if dot(contact normal, Y) <= threshold, the contact is ignored.
    :return: a list of integer object_ids for the set of objects "ontop" of object_a.
    """

    link_id = None
    if isinstance(object_a, int):
        subject_object = get_obj_from_id(sim, object_a)
        if subject_object is None:
            raise AssertionError(
                f"The passed object_id {object_a} is invalid."
            )
        if subject_object.object_id != object_a:
            # object_a is a link
            link_id = subject_object.link_object_ids[object_a]
        object_a = subject_object

    if do_collision_detection:
        sim.perform_discrete_collision_detection()

    yup = mn.Vector3(0.0, 1.0, 0.0)

    ontop_object_ids = []
    for cp in sim.get_physics_contact_points():
        contacting_obj_id = None
        obj_is_b = False
        if cp.object_id_a == object_a.object_id and (
            link_id is None or link_id == cp.link_id_a
        ):
            contacting_obj_id = cp.object_id_b
        elif cp.object_id_b == object_a.object_id and (
            link_id is None or link_id == cp.link_id_b
        ):
            contacting_obj_id = cp.object_id_a
            obj_is_b = True
        if contacting_obj_id is not None:
            contact_normal = (
                cp.contact_normal_on_b_in_ws
                if obj_is_b
                else -cp.contact_normal_on_b_in_ws
            )
            if (
                mn.math.dot(contact_normal, yup)
                > vertical_normal_error_threshold
            ):
                ontop_object_ids.append(contacting_obj_id)

    ontop_object_ids = list(set(ontop_object_ids))

    return ontop_object_ids


def on_floor(
    sim: habitat_sim.Simulator,
    object_a: habitat_sim.physics.ManagedRigidObject,
    distance_threshold: float = 0.04,
    alt_pathfinder: habitat_sim.nav.PathFinder = None,
    island_index: int = -1,
    ao_link_map: Dict[int, int] = None,
) -> bool:
    """
    Checks if the object is heuristically considered to be "on the floor" using the navmesh as an abstraction. This function assumes the PathFinder and parameters provided approximate the navigable floor space well.
    NOTE: alt_pathfinder option can be used to provide an alternative navmesh sized for objects. This would allow objects to be, for example, under tables or in corners and still be considered on the navmesh.

    :param sim: The Simulator instance.
    :param object_a: The object instance.
    :param distance_threshold: Maximum allow-able displacement between current object position and navmesh snapped position.
    :param alt_pathfinder:Optionally provide an alternative PathFinder specifically configured for this check. Defaults to sim.pathfinder.
    :param island_index: Optionally limit allowed navmesh to a specific island. Default (-1) is full navmesh. Note the default is likely not good since large furniture objets could have isolated islands on them which are not the floor.
    :param ao_link_map: A pre-computed map from link object ids to their parent ArticulatedObject's object id.
    :return: Whether or not the object is considered "on the floor" given the configuration.
    """

    assert (
        not object_a.is_articulated
    ), "Object must be ManagedRigidObject, not implemented for ArticulatedObjects or links."

    if alt_pathfinder is None:
        alt_pathfinder = sim.pathfinder

    assert alt_pathfinder.is_loaded

    # use the object's heuristic size to estimate distance from the object center to the navmesh in order to regularize the navigability constraint for larger objects
    obj_size, center = get_obj_size_along(
        sim,
        object_a.object_id,
        mn.Vector3(0.0, -1.0, 0.0),
        ao_link_map=ao_link_map,
    )

    obj_snap = alt_pathfinder.snap_point(center, island_index=island_index)

    # include navmesh cell height error in the distance threshold.
    navmesh_cell_height = alt_pathfinder.nav_mesh_settings.cell_height
    snap_disp = obj_snap - center
    snap_dist = snap_disp.length() - obj_size - (navmesh_cell_height / 2.0)

    return snap_dist <= distance_threshold


def object_in_region(
    sim: habitat_sim.Simulator,
    object_a: Union[
        habitat_sim.physics.ManagedRigidObject,
        habitat_sim.physics.ManagedArticulatedObject,
    ],
    region: habitat_sim.scene.SemanticRegion,
    containment_threshold=0.25,
    center_only=False,
    ao_link_map: Dict[int, int] = None,
) -> Tuple[bool, float]:
    """
    Check if an object is within a region by checking region containment of keypoints.

    :param sim: The Simulator instance.
    :param object_a: The object instance.
    :param region: The SemanticRegion to check.
    :param containment_threshold: threshold ratio of keypoints which need to be in a region to count as containment.
    :param center_only: If True, only use the BB center keypoint, all or nothing.
    :param ao_link_map: A pre-computed map from link object ids to their parent ArticulatedObject's object id.
    :return: boolean containment and the ratio of keypoints which are inside the region.
    """

    key_points = get_global_keypoints_from_object_id(
        sim,
        object_id=object_a.object_id,
        ao_link_map=ao_link_map,
    )

    if center_only:
        key_points = [key_points[0]]

    contained_points = [p for p in key_points if region.contains(p)]
    ratio = len(contained_points) / float(len(key_points))

    return ratio >= containment_threshold, ratio


def get_object_regions(
    sim: habitat_sim.Simulator,
    object_a: Union[
        habitat_sim.physics.ManagedRigidObject,
        habitat_sim.physics.ManagedArticulatedObject,
    ],
    ao_link_map: Dict[int, int] = None,
) -> List[Tuple[int, float]]:
    """
    Get a sorted list of regions containing an object using bounding box keypoints.

    :param sim: The Simulator instance.
    :param object_a: The object instance.
    :param ao_link_map: A pre-computed map from link object ids to their parent ArticulatedObject's object id.
    :return: A sorted list of region index, ratio pairs. First item in the list the primary containing region.
    """

    key_points = get_global_keypoints_from_object_id(
        sim,
        object_id=object_a.object_id,
        ao_link_map=ao_link_map,
    )

    return sim.semantic_scene.get_regions_for_points(key_points)


def get_floor_point_in_region(
    sim: habitat_sim.Simulator,
    region_index: int,
    island_index: int = -1,
    max_center_samples: int = 100,
    max_global_samples: int = 1000,
    quick_return: bool = False,
) -> Optional[mn.Vector3]:
    """
    Sample the navmesh to find a point on the floor within a given region.

    :param sim: The Simulator instance.
    :param region_index: The index of the Region within which to sample.
    :param island_index: The index of the navmesh island representing the active floor area. Default -1 is all islands. Should be set to the same island used for other navmesh operations in the application. For example, the largest indoor island for the scene.
    :param max_center_samples: The number of samples near the center point to attempt if applicable. This will be done first. <=0 skips center sampling.
    :param max_global_samples: The number of global navmesh samples to attempt if center point samples were unsuccessful. <=0 skips this step.
    :param quick_return: If True, the first valid sample will be returned instead of continuing to search for a better sample. Use this option when speed is more important than the quality or consistency.
    :return: The sampled floor point within the given region or None if a point could not be found.

    This method attempts to find a point in the region with maximum navmesh clearance by sorting candidates on `distance_to_closest_obstacle`.
    Because this method uses multiple sampling passes it is advised to use it in initialization and pre-processes rather than within an application loop.
    """

    # get the SemanticRegion from the index
    region = sim.semantic_scene.regions[region_index]

    #################
    # sampling points:
    attempts = 0
    best_sample: Optional[mn.Vector3] = None
    best_navmesh_dist = -1

    # first try aiming at the center (nice for convex regions)
    if max_center_samples > 0:
        # get the center of the region's bounds and attempt to snap it to the navmesh
        region_center = region.aabb.center()
        region_center_snap = sim.pathfinder.snap_point(
            region_center, island_index=island_index
        )

        if not np.isnan(region_center_snap[0]):
            # sampling near the center
            while attempts < max_center_samples:
                # get a point within 1 meter of the snapped region center if possible
                sample = sim.pathfinder.get_random_navigable_point_near(
                    region_center_snap, radius=1.0, island_index=island_index
                )
                if not np.isnan(sample[0]) and region.contains(sample):
                    navmesh_dist = sim.pathfinder.distance_to_closest_obstacle(
                        sample
                    )
                    if navmesh_dist > best_navmesh_dist:
                        # found a valid point in a more "open" part of the region
                        best_sample = sample
                        best_navmesh_dist = navmesh_dist
                        if quick_return:
                            # short-circuit to return the first valid sample
                            return best_sample
                attempts += 1
        else:
            # region center doesn't snap, so move on
            pass

    # try again without aiming for the center (in case of concave region)
    if best_sample is None:
        attempts = 0
        while attempts < max_global_samples:
            sample = sim.pathfinder.get_random_navigable_point(
                island_index=island_index
            )
            if region.contains(sample):
                navmesh_dist = sim.pathfinder.distance_to_closest_obstacle(
                    sample
                )
                if navmesh_dist > best_navmesh_dist:
                    # found a valid point in a more "open" part of the region
                    best_sample = sample
                    best_navmesh_dist = navmesh_dist
                    if quick_return:
                        # short-circuit to return the first valid sample
                        return best_sample
            attempts += 1

    return best_sample


def get_link_normalized_joint_position(
    object_a: habitat_sim.physics.ManagedArticulatedObject, link_ix: int
) -> float:
    """
    Normalize the joint limit range [min, max] -> [0,1] and return the current joint state in this range.

    :param object_a: The parent ArticulatedObject of the link.
    :param link_ix: The index of the link within the parent object. Not the link's object_id.
    :return: normalized joint position [0,1]
    """

    assert object_a.get_link_joint_type(link_ix) in [
        habitat_sim.physics.JointType.Revolute,
        habitat_sim.physics.JointType.Prismatic,
    ], f"Invalid joint type '{object_a.get_link_joint_type(link_ix)}'. Open/closed not a valid check for multi-dimensional or fixed joints."

    joint_pos_ix = object_a.get_link_joint_pos_offset(link_ix)
    joint_pos = object_a.joint_positions[joint_pos_ix]
    limits = object_a.joint_position_limits

    # compute the normalized position [0,1]
    n_pos = (joint_pos - limits[0][joint_pos_ix]) / (
        limits[1][joint_pos_ix] - limits[0][joint_pos_ix]
    )
    return n_pos


def set_link_normalized_joint_position(
    object_a: habitat_sim.physics.ManagedArticulatedObject,
    link_ix: int,
    normalized_pos: float,
) -> None:
    """
    Set the joint's state within its limits from a normalized range [0,1] -> [min, max]

    :param object_a: The parent ArticulatedObject of the link.
    :param link_ix: The index of the link within the parent object. Not the link's object_id.
    :param normalized_pos: The normalized position [0,1] to set.

    Assumes the joint has valid joint limits.
    """

    assert object_a.get_link_joint_type(link_ix) in [
        habitat_sim.physics.JointType.Revolute,
        habitat_sim.physics.JointType.Prismatic,
    ], f"Invalid joint type '{object_a.get_link_joint_type(link_ix)}'. Open/closed not a valid check for multi-dimensional or fixed joints."

    assert (
        normalized_pos <= 1.0 and normalized_pos >= 0
    ), "values outside the range [0,1] are by definition beyond the joint limits."

    joint_pos_ix = object_a.get_link_joint_pos_offset(link_ix)
    limits = object_a.joint_position_limits
    joint_positions = object_a.joint_positions
    joint_positions[joint_pos_ix] = limits[0][joint_pos_ix] + (
        normalized_pos * (limits[1][joint_pos_ix] - limits[0][joint_pos_ix])
    )
    object_a.joint_positions = joint_positions


def link_is_open(
    object_a: habitat_sim.physics.ManagedArticulatedObject,
    link_ix: int,
    threshold: float = 0.4,
) -> bool:
    """
    Check whether a particular AO link is in the "open" state.
    We assume that joint limits define the closed state (min) and open state (max).

    :param object_a: The parent ArticulatedObject of the link to check.
    :param link_ix: The index of the link within the parent object. Not the link's object_id.
    :param threshold: The normalized threshold ratio of joint ranges which are considered "open". E.g. 0.8 = 80%
    :return: Whether or not the link is considered "open".
    """

    return get_link_normalized_joint_position(object_a, link_ix) >= threshold


def link_is_closed(
    object_a: habitat_sim.physics.ManagedArticulatedObject,
    link_ix: int,
    threshold: float = 0.1,
) -> bool:
    """
    Check whether a particular AO link is in the "closed" state.
    We assume that joint limits define the closed state (min) and open state (max).

    :param object_a: The parent ArticulatedObject of the link to check.
    :param link_ix: The index of the link within the parent object. Not the link's object_id.
    :param threshold: The normalized threshold ratio of joint ranges which are considered "closed". E.g. 0.1 = 10%
    :return: Whether or not the link is considered "closed".
    """

    return get_link_normalized_joint_position(object_a, link_ix) <= threshold


def close_link(
    object_a: habitat_sim.physics.ManagedArticulatedObject, link_ix: int
) -> None:
    """
    Set a link to the "closed" state. Sets the joint position to the minimum joint limit.

    TODO: does not do any collision checking to validate the state or move any other objects which may be contained in or supported by this link.

    :param object_a: The parent ArticulatedObject of the link to check.
    :param link_ix: The index of the link within the parent object. Not the link's object_id.
    """

    set_link_normalized_joint_position(object_a, link_ix, 0)


def open_link(
    object_a: habitat_sim.physics.ManagedArticulatedObject, link_ix: int
) -> None:
    """
    Set a link to the "open" state. Sets the joint position to the maximum joint limit.

    :param object_a: The parent ArticulatedObject of the link to check.
    :param link_ix: The index of the link within the parent object. Not the link's object_id.

    TODO: does not do any collision checking to validate the state or move any other objects which may be contained in or supported by this link.
    """

    set_link_normalized_joint_position(object_a, link_ix, 1.0)


def bb_next_to(
    bb_a: mn.Range3D,
    bb_b: mn.Range3D,
    transform_a: mn.Matrix4 = None,
    transform_b: mn.Matrix4 = None,
    hor_l2_threshold: float = 0.3,
    vertical_padding: float = 0.1,
) -> bool:
    """
    Check whether or not two bounding boxes should be considered "next to" one another.

    :param bb_a: local bounding box of one object
    :param bb_b: local bounding box of another object
    :param transform_a: local to global transform for the first object. Default is identity.
    :param transform_b: local to global transform for the second object. Default is identity.
    :param hor_l2_threshold: regularized horizontal L2 distance allowed between the objects' centers.
    :param vertical_padding: vertical distance padding used when comparing bounding boxes. Higher value is a looser constraint.
    :return: Whether or not the objects are heuristically "next to" one another.

    Concretely, consists of two checks:
        1. assert overlap between the vertical range of the two bounding boxes.

        2. regularized horizontal L2 distance between object centers. Regularized in this case means projected displacement vector is truncated by each object's heuristic size.

    """

    if transform_a is None:
        transform_a = mn.Matrix4.identity_init()
    if transform_b is None:
        transform_b = mn.Matrix4.identity_init()

    keypoints_a = get_global_keypoints_from_bb(bb_a, transform_a)
    keypoints_b = get_global_keypoints_from_bb(bb_b, transform_b)

    lowest_height_a = min([p[1] for p in keypoints_a])
    lowest_height_b = min([p[1] for p in keypoints_b])
    highest_height_a = max([p[1] for p in keypoints_a])
    highest_height_b = max([p[1] for p in keypoints_b])

    # check for non-overlapping bounding boxes
    if (highest_height_a + vertical_padding) < lowest_height_b or (
        highest_height_b + vertical_padding
    ) < lowest_height_a:
        return False

    if (
        size_regularized_bb_distance(
            bb_a, bb_b, transform_a, transform_b, flatten_axis=1
        )
        > hor_l2_threshold
    ):
        return False

    return True


def obj_next_to(
    sim: habitat_sim.Simulator,
    object_id_a: int,
    object_id_b: int,
    hor_l2_threshold: float = 0.5,
    vertical_padding: float = 0.1,
    ao_link_map: Dict[int, int] = None,
) -> bool:
    """
    Check whether or not two objects should be considered "next to" one another.


    :param sim: The Simulator instance.
    :param object_id_a: object_id of the first ManagedObject or link.
    :param object_id_b: object_id of the second ManagedObject or link.
    :param hor_l2_threshold: regularized horizontal L2 distance allow between the objects' centers. This should be tailored to the scenario.
    :param vertical_padding: vertical distance padding added when comparing the object's bounding boxes. Higher value is a looser constraint.
    :param ao_link_map: A pre-computed map from link object ids to their parent ArticulatedObject's object id.
    :return: Whether or not the objects are heuristically "next to" one another.

    Concretely, consists of two checks:
        1. bounding boxes must overlap vertically.
        2. regularized horizontal L2 distance between object centers must be less than a threshold. Regularized in this case means displacement vector is truncated by each object's heuristic size.
    """

    assert object_id_a != object_id_b, "Object cannot be 'next to' itself."

    assert (
        object_id_a != habitat_sim.stage_id
        and object_id_b != habitat_sim.stage_id
    ), "Cannot compute distance between the stage and its contents."

    obja_bb, transform_a = get_bb_for_object_id(sim, object_id_a, ao_link_map)
    objb_bb, transform_b = get_bb_for_object_id(sim, object_id_b, ao_link_map)

    return bb_next_to(
        obja_bb,
        objb_bb,
        transform_a,
        transform_b,
        hor_l2_threshold,
        vertical_padding,
    )


def point_to_tri_dist(
    point: np.ndarray, triangles: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Compute the minimum distance between a 3D point and a set of triangles (e.g. a triangle mesh) and return both the minimum distance and that closest point.
    Uses vectorized numpy operations for high performance with a large number of triangles.
    Implementation adapted from https://stackoverflow.com/questions/32342620/closest-point-projection-of-a-3d-point-to-3d-triangles-with-numpy-scipy
    Algorithm is vectorized form of e.g. https://www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf

    :param point: A 3D point.
    :param triangles: An nx3x3 numpy array of triangles. Each entry of the first axis is a triangle with three 3D vectors, the vertices of the triangle.
    :return: The minimum distance from point to triangle set and the closest point on the surface of any triangle.
    """

    with np.errstate(all="ignore"):
        # Unpack triangle points
        p0, p1, p2 = np.asarray(triangles).swapaxes(0, 1)

        # Calculate triangle edges
        e0 = p1 - p0
        e1 = p2 - p0
        a = np.einsum("...i,...i", e0, e0)
        b = np.einsum("...i,...i", e0, e1)
        c = np.einsum("...i,...i", e1, e1)

        # Calculate determinant and denominator
        det = a * c - b * b
        invDet = 1.0 / det
        denom = a - 2 * b + c

        # Project to the edges
        p = p0 - point
        d = np.einsum("...i,...i", e0, p)
        e = np.einsum("...i,...i", e1, p)
        u = b * e - c * d
        v = b * d - a * e

        # Calculate numerators
        bd = b + d
        ce = c + e
        numer0 = (ce - bd) / denom
        numer1 = (c + e - b - d) / denom
        da = -d / a
        ec = -e / c

        # Vectorize test conditions
        m0 = u + v < det
        m1 = u < 0
        m2 = v < 0
        m3 = d < 0
        m4 = a + d > b + e
        m5 = ce > bd

        t0 = m0 & m1 & m2 & m3
        t1 = m0 & m1 & m2 & ~m3
        t2 = m0 & m1 & ~m2
        t3 = m0 & ~m1 & m2
        t4 = m0 & ~m1 & ~m2
        t5 = ~m0 & m1 & m5
        t6 = ~m0 & m1 & ~m5
        t7 = ~m0 & m2 & m4
        t8 = ~m0 & m2 & ~m4
        t9 = ~m0 & ~m1 & ~m2

        u = np.where(t0, np.clip(da, 0, 1), u)
        v = np.where(t0, 0, v)
        u = np.where(t1, 0, u)
        v = np.where(t1, 0, v)
        u = np.where(t2, 0, u)
        v = np.where(t2, np.clip(ec, 0, 1), v)
        u = np.where(t3, np.clip(da, 0, 1), u)
        v = np.where(t3, 0, v)
        u *= np.where(t4, invDet, 1)
        v *= np.where(t4, invDet, 1)
        u = np.where(t5, np.clip(numer0, 0, 1), u)
        v = np.where(t5, 1 - u, v)
        u = np.where(t6, 0, u)
        v = np.where(t6, 1, v)
        u = np.where(t7, np.clip(numer1, 0, 1), u)
        v = np.where(t7, 1 - u, v)
        u = np.where(t8, 1, u)
        v = np.where(t8, 0, v)
        u = np.where(t9, np.clip(numer1, 0, 1), u)
        v = np.where(t9, 1 - u, v)
        u = u[:, None]
        v = v[:, None]

        # this array contains a list of points, the closest on each triangle
        closest_points_each_tri = p0 + u * e0 + v * e1

        # now extract the closest point on the mesh and minimum distance for return
        closest_point_index = np.argmin(
            spatial.distance.cdist(np.array([point]), closest_points_each_tri),
            axis=1,
        )
        closest_point: np.ndarray = closest_points_each_tri[
            closest_point_index
        ]
        min_dist = float(np.linalg.norm(point - closest_point))

        # Return the minimum distance
        return min_dist, closest_point


def match_point_to_receptacle(
    point: mn.Vector3,
    sim: habitat_sim.Simulator,
    candidate_receptacles: List["Receptacle"],
    max_dist_to_rec: float = 0.25,
) -> Tuple[List[str], float, str]:
    """
    Heuristic to match a 3D point with the nearest Receptacle(s).

    :param sim: The Simulator instance.
    :param candidate_receptacles: a list of candidate Receptacles for matching.
    :param max_dist_to_rec: The threshold point to mesh distance to be matched with a Receptacle.
    :return: Tuple containing: (1): list of receptacle strings [Receptacle.unique_name] or an empty list (2): a floating point confidence score [0,1] (3): a message string describing the results for use in a UI tooltip
    """

    # get point to rec distances for the candidates
    dist_to_recs = [
        rec.dist_to_rec(sim, point) for rec in candidate_receptacles
    ]

    # define an epsilon distance similarity threshold for determining two receptacles are equidistant (e.g. overlapping)
    same_dist_eps = 0.01
    min_indices = []
    min_dist: float = None
    for ix, dist in enumerate(dist_to_recs):
        if dist < max_dist_to_rec:
            if min_dist is None or dist < (min_dist - same_dist_eps):
                min_dist = dist
                min_indices = [ix]
            elif abs(dist - min_dist) < same_dist_eps:
                # this rec is nearly equidistant to the current best match
                min_indices.append(ix)

    # if match(es) found, return it/them
    if len(min_indices) > 0:
        return (
            [candidate_receptacles[ix].unique_name for ix in min_indices],
            1.0 - (min_dist / max_dist_to_rec),
            "successful match",
        )

    # default empty return for failure
    return ([], 0, "No match: point is too far from a candidate Receptacle.")


def get_obj_receptacle_and_confidence(
    sim: habitat_sim.Simulator,
    obj: habitat_sim.physics.ManagedRigidObject,
    candidate_receptacles: Dict[str, "Receptacle"],
    support_surface_id: Optional[int] = None,
    obj_bottom_location: Optional[mn.Vector3] = None,
    max_dist_to_rec: float = 0.25,
    island_index: int = -1,
) -> Tuple[List[str], float, str]:
    """
    Heuristic to match a potential placement point with a Receptacle and provide some confidence.

    :param sim: The Simulator instance.
    :param obj: The ManagedRigidObject for which to find a Receptacle match.
    :param candidate_receptacles: a dict (unique_name to Receptacle) of candidate Receptacles for matching.
    :param support_surface_id: The object_id of the intended support surface (rigid object, articulated link, or stage_id). If not provided, a downward raycast from object center will determine the first hit as the support surface.
    :param obj_bottom_location: The optional location of the candidate bottom point of the object. If not provided, project the object center to the lowest extent.
    :param max_dist_to_rec: The threshold point to mesh distance to be matched with a Receptacle.
    :param island_index: Optionally provide an island_index for identifying navigable floor points. Default is full navmesh.
    :return: Tuple containing: (1): list of receptacle strings: Receptacle.unique_name, "floor,<region.id>", "<region.id>" or an empty list (2): a floating point confidence score [0,1] (3): a message string describing the results for use in a UI tooltip

    When using this util for candidate placement with raycasting (e.g. HitL): provide 'support_surface_id' and 'obj_bottom_location' overrides from the raycast.
    When using this util for assessing current object state (e.g. episode evaluation): leave 'support_surface_id' and 'obj_bottom_location' as default.
    """
    # collect information about failure of the function
    info_text = ""

    # get the center point of the object projected on the local bounding box size in the gravity direction
    grav_vector = sim.get_gravity().normalized()
    dist, center = get_obj_size_along(sim, obj.object_id, grav_vector)
    # either compute or use the provided object location
    obj_bottom_point = (
        center + grav_vector * dist
        if obj_bottom_location is None
        else obj_bottom_location
    )

    # find a support surface if one was not provided
    if support_surface_id is None:
        raycast_results = sim.cast_ray(
            habitat_sim.geo.Ray(center, grav_vector)
        )
        if raycast_results.has_hits():
            for hit in raycast_results.hits:
                if hit.object_id != obj.object_id:
                    # get the first ray hit against a different object
                    support_surface_id = hit.object_id
                    break
        if support_surface_id is None:
            info_text = "No support surface found for object."
            return [], 1.0, info_text

    # in case we cannot match the object|point to a Receptacle object, we'll try to validate that it could be "on the floor"
    fallback_to_floor_matching: bool = False

    if support_surface_id == habitat_sim.stage_id:
        # support_surface on stage could be the floor
        fallback_to_floor_matching = True
    else:
        support_object = get_obj_from_id(sim, support_surface_id)
        matching_recs = [
            rec
            for u_name, rec in candidate_receptacles.items()
            if support_object.handle in u_name
        ]
        if support_object.object_id != support_surface_id:
            # support object is a link
            link_index = support_object.link_object_ids[support_surface_id]
            # further cull the list to this link's recs
            matching_recs = [
                rec for rec in matching_recs if rec.parent_link == link_index
            ]
        if len(matching_recs) == 0:
            # there are no Receptacles for this support surface
            fallback_to_floor_matching = True
        else:
            # try matching to the candidate receptacles
            matches, confidence, info_text = match_point_to_receptacle(
                obj_bottom_point,
                sim,
                matching_recs,
                max_dist_to_rec=max_dist_to_rec,
            )
            if len(matches) > 0:
                return (matches, confidence, info_text)

    # map the point to regions
    point_regions = sim.semantic_scene.get_weighted_regions_for_point(
        obj_bottom_point
    )

    # If we made it this far, matching to a Receptacle object failed.
    # Now for some cases, check if the point is navigable and if so, try matching it to a region
    if fallback_to_floor_matching:
        # NOTE: using navmesh snapping to a point within 10cm horizontally as a heuristic for "on the floor"
        snap_point = sim.pathfinder.snap_point(obj_bottom_point, island_index)
        if (obj_bottom_point - snap_point).length() < 0.15:
            # this point is on the floor
            if len(point_regions) > 0:
                # found matching regions, pick the primary (most precise) one
                region_name = sim.semantic_scene.regions[
                    point_regions[0][0]
                ].id
            else:
                # point is not matched to a region
                region_name = "unknown_region"
            return [f"floor,{region_name}"], 1.0, "successful match"
        else:
            info_text = (
                "Point does not match any Receptacle and is not navigable."
            )

    # not on the floor or matched to a receptacle, so return the region.id
    if len(point_regions) > 0:
        return (
            [sim.semantic_scene.regions[point_regions[0][0]].id],
            1.0,
            "region match",
        )

    # all receptacles are too far away or there are no matches
    return [], 1.0, info_text
