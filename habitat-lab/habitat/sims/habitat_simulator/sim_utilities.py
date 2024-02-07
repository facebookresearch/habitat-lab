#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import magnum as mn

import habitat_sim
from habitat.sims.habitat_simulator.debug_visualizer import DebugVisualizer


def register_custom_wireframe_box_template(
    sim: habitat_sim.Simulator,
    size: mn.Vector3,
    template_name: str = "custom_wireframe_box",
) -> str:
    """
    Generate and register a custom template for a wireframe box of given size.
    Return the new template's handle.
    """
    obj_attr_mgr = sim.get_object_template_manager()
    cube_template = obj_attr_mgr.get_template_by_handle(
        obj_attr_mgr.get_template_handles("cubeWireframe")[0]
    )
    cube_template.scale = size
    obj_attr_mgr.register_template(cube_template, template_name)
    return template_name


def add_wire_box(
    sim: habitat_sim.Simulator,
    size: mn.Vector3,
    center: mn.Vector3,
    attach_to: Optional[habitat_sim.scene.SceneNode] = None,
    orientation: Optional[mn.Quaternion] = None,
) -> habitat_sim.physics.ManagedRigidObject:
    """
    Generate a wire box object and optionally attach it to another existing object (automatically applies object scale).
    Returns the new object.
    """
    if orientation is None:
        orientation = mn.Quaternion()
    box_template_handle = register_custom_wireframe_box_template(sim, size)
    new_object = sim.get_rigid_object_manager().add_object_by_template_handle(
        box_template_handle, attach_to
    )
    new_object.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    new_object.collidable = False
    # translate to local offset if attached or global offset if not
    new_object.translation = center
    new_object.rotation = orientation
    return new_object


def add_transformed_wire_box(
    sim: habitat_sim.Simulator,
    size: mn.Vector3,
    transform: Optional[mn.Matrix4] = None,
) -> habitat_sim.physics.ManagedRigidObject:
    """
    Generate a transformed wire box in world space.
    Returns the new object.
    """
    if transform is None:
        transform = mn.Matrix4()
    box_template_handle = register_custom_wireframe_box_template(sim, size)
    new_object = sim.get_rigid_object_manager().add_object_by_template_handle(
        box_template_handle
    )
    new_object.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    new_object.collidable = False
    # translate to local offset if attached or global offset if not
    new_object.transformation = transform
    return new_object


def add_viz_sphere(
    sim: habitat_sim.Simulator, radius: float, pos: mn.Vector3
) -> habitat_sim.physics.ManagedRigidObject:
    """
    Add a visualization-only sphere to the world at a global position.
    Returns the new object.
    """
    obj_attr_mgr = sim.get_object_template_manager()
    sphere_template = obj_attr_mgr.get_template_by_handle(
        obj_attr_mgr.get_template_handles("icosphereWireframe")[0]
    )
    sphere_template.scale = mn.Vector3(radius)
    obj_attr_mgr.register_template(sphere_template, "viz_sphere")
    new_object = sim.get_rigid_object_manager().add_object_by_template_handle(
        "viz_sphere"
    )
    new_object.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    new_object.collidable = False
    new_object.translation = pos
    return new_object


def get_bb_corners(range3d: mn.Range3D) -> List[mn.Vector3]:
    """
    Return a list of AABB (Range3D) corners in object local space.
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


def get_ao_global_bb(
    obj: habitat_sim.physics.ManagedArticulatedObject,
) -> Optional[mn.Range3D]:
    """
    Compute the cumulative bounding box of an ArticulatedObject by merging all link bounding boxes.
    """

    cumulative_global_bb: mn.Range3D = None
    for link_ix in range(-1, obj.num_links):
        link_node = obj.get_link_scene_node(link_ix)
        bb = link_node.cumulative_bb
        global_bb = habitat_sim.geo.get_transformed_bb(
            bb, link_node.absolute_transformation()
        )
        if cumulative_global_bb is None:
            cumulative_global_bb = global_bb
        else:
            cumulative_global_bb = mn.math.join(
                cumulative_global_bb, global_bb
            )
    return cumulative_global_bb


def bb_ray_prescreen(
    sim: habitat_sim.Simulator,
    obj: habitat_sim.physics.ManagedRigidObject,
    support_obj_ids: Optional[List[int]] = None,
    check_all_corners: bool = False,
) -> Dict[str, Any]:
    """
    Pre-screen a potential placement by casting rays in the gravity direction from the object center of mass (and optionally each corner of its bounding box) checking for interferring objects below.

    :param sim: The Simulator instance.
    :param obj: The RigidObject instance.
    :param support_obj_ids: A list of object ids designated as valid support surfaces for object placement. Contact with other objects is a criteria for placement rejection.
    :param check_all_corners: Optionally cast rays from all bounding box corners instead of only casting a ray from the center of mass.
    """
    if support_obj_ids is None:
        # set default support surface to stage/ground mesh
        support_obj_ids = [-1]
    lowest_key_point: mn.Vector3 = None
    lowest_key_point_height = None
    highest_support_impact: Optional[mn.Vector3] = None
    highest_support_impact_height = None
    highest_support_impact_id = None
    raycast_results = []
    gravity_dir = sim.get_gravity().normalized()
    object_local_to_global = obj.transformation
    bb_corners = get_bb_corners(obj.root_scene_node.cumulative_bb)
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
                if hit.object_id == obj.object_id:
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
    elif highest_support_impact_id == -1:
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
    dbv: Optional[DebugVisualizer] = None,
) -> bool:
    """
    Attempt to project an object in the gravity direction onto the surface below it.

    :param sim: The Simulator instance.
    :param obj: The RigidObject instance.
    :param support_obj_ids: A list of object ids designated as valid support surfaces for object placement. Contact with other objects is a criteria for placement rejection. If none provided, default support surface is the stage/ground mesh (-1).
    :param dbv: Optionally provide a DebugVisualizer (dbv) to render debug images of each object's computed snap position before collision culling.

    Reject invalid placements by checking for penetration with other existing objects.
    Returns boolean success.
    If placement is successful, the object state is updated to the snapped location.
    If placement is rejected, object position is not modified and False is returned.

    To use this utility, generate an initial placement for any object above any of the designated support surfaces and call this function to attempt to snap it onto the nearest surface in the gravity direction.
    """
    cached_position = obj.translation

    if support_obj_ids is None:
        # set default support surface to stage/ground mesh
        support_obj_ids = [-1]

    bb_ray_prescreen_results = bb_ray_prescreen(
        sim, obj, support_obj_ids, check_all_corners=False
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
                cp.object_id_a == obj.object_id
                or cp.object_id_b == obj.object_id
            ) and (
                (cp.contact_distance < -0.01)
                or not (
                    cp.object_id_a in support_obj_ids
                    or cp.object_id_b in support_obj_ids
                )
            ):
                obj.translation = cached_position
                # print(f" Failure: contact in final position w/ distance = {cp.contact_distance}.")
                # print(f" Failure: contact in final position with non support object {cp.object_id_a} or {cp.object_id_b}.")
                return False
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
    """
    managers = [
        sim.get_rigid_object_manager(),
        sim.get_articulated_object_manager(),
    ]
    all_objects = []
    for mngr in managers:
        all_objects.extend(mngr.get_objects_by_handle_substring().values())
    return all_objects


def get_obj_size_along(
    sim: habitat_sim.Simulator,
    object_id: int,
    global_vec: mn.Vector3,
    ao_link_map: Dict[int, int] = None,
    ao_aabbs: Dict[int, mn.Range3D] = None,
) -> float:
    """
    Uses object bounding box as a heuristic to estimate object size in a particular global direction.
    """
    obj_bb, transform = get_bb_for_object_id(
        sim, object_id, ao_link_map, ao_aabbs
    )
    local_scale = mn.Matrix4.scaling(obj_bb.size() / 2.0)
    local_vec = transform.inverted().transform_vector(global_vec)
    local_vec_size = local_scale.transform_vector(local_vec).length()
    return local_vec_size


def size_regularized_distance(
    sim: habitat_sim.Simulator,
    objectA,
    objectB,
    ao_link_map: Dict[int, int] = None,
    ao_aabbs: Dict[int, mn.Range3D] = None,
) -> float:
    """
    Get the distance between two objects regularized by their size.
    """
    obja_bb, transform_a = get_bb_for_object_id(
        sim, objectA.object_id, ao_link_map, ao_aabbs
    )
    objb_bb, transform_b = get_bb_for_object_id(
        sim, objectB.object_id, ao_link_map, ao_aabbs
    )

    a_center = transform_a.transform_point(obja_bb.center())
    b_center = transform_b.transform_point(objb_bb.center())

    disp = a_center - b_center
    dist = disp.length()
    disp_dir = disp / dist

    local_scale_a = mn.Matrix4.scaling(obja_bb.size() / 2.0)
    local_vec_a = transform_a.inverted().transform_vector(disp_dir)
    local_vec_size_a = local_scale_a.transform_vector(local_vec_a).length()

    local_scale_b = mn.Matrix4.scaling(objb_bb.size() / 2.0)
    local_vec_b = transform_b.inverted().transform_vector(disp_dir)
    local_vec_size_b = local_scale_b.transform_vector(local_vec_b).length()

    # if object bounding boxes are significantly overlapping then distance may be negative, clamp to 0
    return max(0, dist - local_vec_size_a - local_vec_size_b)


# ============================================================
# New Sim Query Utils
# ============================================================


def get_bb_for_object_id(
    sim: habitat_sim.Simulator,
    obj_id: int,
    ao_link_map: Dict[int, int] = None,
    ao_aabbs: Dict[int, mn.Range3D] = None,
) -> Tuple[mn.Range3D, mn.Matrix4]:
    """
    Wrapper to get a bb and global transform directly from an object id.
    Handles RigidObject and ArticulatedLink ids.
    TODO: Handle ArticulatedObject root bounding boxes

    :param sim: The Simulator instance.
    :param obj_id: The integer id of the object or link.
    :param ao_link_map: A pre-computed map from link object ids to their parent ArticulatedObject's object id.
    :param ao_aabbs: A pre-computed map from ArticulatedObject object_ids to their local bounding boxes. If not provided, recomputed as necessary.

    :return: tuple (local_aabb, global_transform)
    """

    if ao_link_map is None:
        # Note: better to pre-compute this and pass it around
        ao_link_map = get_ao_link_id_map(sim)

    # check for a link
    if obj_id in ao_link_map:
        ao = sim.get_articulated_object_manager().get_object_by_id(
            ao_link_map[obj_id]
        )
        if ao.object_id == obj_id:
            ao_aabb = None
            # This is the AO body
            if ao_aabbs is None:
                ao_aabb = get_ao_root_bb(ao)
            else:
                ao_aabb = ao_aabbs[obj_id]
            return (ao_aabb, ao.transformation)
        else:
            link_node = ao.get_link_scene_node(ao.link_object_ids[obj_id])
            link_transform = link_node.absolute_transformation()
            return (link_node.cumulative_bb, link_transform)
    rom = sim.get_rigid_object_manager()
    if rom.get_library_has_id(obj_id):
        ro = rom.get_object_by_id(obj_id)
        return (ro.root_scene_node.cumulative_bb, ro.transformation)
    raise AssertionError("obj_id not found, this is unexpected.")


def get_ao_root_bb(
    ao: habitat_sim.physics.ManagedArticulatedObject,
) -> mn.Range3D:
    """
    Get the local bounding box of all links of an articulated object in the root frame.

    :param ao: The ArticulatedObject instance.
    """

    ao_local_part_bb_corners = []
    # NOTE: this is empty because the links are not in the subtree of the root
    # ao.root_scene_node.compute_cumulative_bb()
    # print(f"ao {ao.handle} cumulative_bb = {ao.root_scene_node.cumulative_bb}")
    # print(f"ao {ao.handle} cumulative_bb should be = {ao.root_scene_node.compute_cumulative_bb()}")
    link_nodes = [ao.get_link_scene_node(ix) for ix in range(-1, ao.num_links)]
    for link_node in link_nodes:
        # print(f"    - link cumulative_bb = {link_node.cumulative_bb}")
        local_bb_corners = get_bb_corners(link_node.cumulative_bb)
        global_bb_corners = [
            link_node.absolute_transformation().transform_point(bb_corner)
            for bb_corner in local_bb_corners
        ]
        ao_local_bb_corners = [
            ao.transformation.inverted().transform_point(p)
            for p in global_bb_corners
        ]
        ao_local_part_bb_corners.extend(ao_local_bb_corners)

    # get min and max of each dimension
    # TODO: use numpy arrays for more elegance...
    max_vec = mn.Vector3(ao_local_part_bb_corners[0])
    min_vec = mn.Vector3(ao_local_part_bb_corners[0])
    for point in ao_local_part_bb_corners:
        for dim in range(3):
            max_vec[dim] = max(max_vec[dim], point[dim])
            min_vec[dim] = min(min_vec[dim], point[dim])
    return mn.Range3D(min_vec, max_vec)


def get_ao_root_bbs(
    sim: habitat_sim.Simulator,
) -> Dict[int, mn.Range3D]:
    """
    Computes a dictionary mapping AO handles to a global bounding box of parts.
    Must be updated when AO state changes to correctly bound the full set of links.

    :param sim: The Simulator instance.

    :return: dictionary mapping ArticulatedObjects' object_id to their bounding box in local space.
    """

    ao_local_bbs: Dict[
        habitat_sim.physics.ManagedBulletArticulatedObject, mn.Range3D
    ] = {}
    aom = sim.get_articulated_object_manager()
    for ao in aom.get_objects_by_handle_substring().values():
        ao_local_bbs[ao.object_id] = get_ao_root_bb(ao)
    return ao_local_bbs


# Prepositional Logic Functions:
def get_ao_link_id_map(sim: habitat_sim.Simulator) -> Dict[int, int]:
    """
    Construct a map of ao_link object ids to their parent ao's object id.
    NOTE: also maps ao's root object id to itself for ease of use.

    :param sim: The Simulator instance.

    :return: dictionary mapping ArticulatedLink object ids to their parent's object id.
    """

    aom = sim.get_articulated_object_manager()
    ao_link_map: Dict[int, int] = {}
    for ao in aom.get_objects_by_handle_substring().values():
        # add the ao itself for ease of use
        ao_link_map[ao.object_id] = ao.object_id
        # add the links
        for link_id in ao.link_object_ids:
            ao_link_map[link_id] = ao.object_id

    # print(f"ao_link_map = {ao_link_map}")

    return ao_link_map


def get_object_global_keypoints(
    objectA: habitat_sim.physics.ManagedRigidObject,
) -> List[mn.Vector3]:
    """
    Get a list of object keypoints in global space.
    0th point is the center of mass (CoM), others are bounding box corners.

    :param objectA: The ManagedRigidObject from which to extract keypoints.

    :return: A set of global 3D keypoints for the object.
    """

    local_keypoints = [mn.Vector3(0)]
    local_keypoints.extend(
        get_bb_corners(objectA.root_scene_node.cumulative_bb)
    )
    global_keypoints = [
        objectA.transformation.transform_point(key_point)
        for key_point in local_keypoints
    ]
    return global_keypoints


def object_keypoint_cast(
    sim: habitat_sim.Simulator,
    objectA: habitat_sim.physics.ManagedRigidObject,
    direction: mn.Vector3 = None,
) -> List[habitat_sim.physics.RaycastResults]:
    """
    Compute's object global keypoints, casts rays from each in the specified direction and returns the resulting RaycastResults.
    Index 0 in the list is the CoM, others are corners.

    :param sim: The Simulator instance.
    :param objectA: The ManagedRigidObject from which to extract keypoints and raycast.
    :param direction: Optionally provide a unit length global direction vector for the raycast. If None, default to -Y.

    :return: A list of RaycastResults, one from each object keypoint.
    """

    if direction is None:
        # default to downward raycast
        direction = mn.Vector3(0, -1, 0)

    global_keypoints = get_object_global_keypoints(objectA)
    return [
        sim.cast_ray(habitat_sim.geo.Ray(keypoint, direction))
        for keypoint in global_keypoints
    ]


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

    ArticulatedLink object_ids will return the ManagedArticulatedObject.
    If you want link id, use ManagedArticulatedObject.link_object_ids[obj_id].

    :param sim: The Simulator instance.
    :param obj_id: object id for which ManagedObject is desired.
    :param ao_link_map: A pre-computed map from link object ids to their parent ArticulatedObject's object id.

    :return: a ManagedObject or None
    """

    if ao_link_map is None:
        # Note: better to pre-compute this and pass it around
        ao_link_map = get_ao_link_id_map(sim)

    rom = sim.get_rigid_object_manager()
    if rom.get_library_has_id(obj_id):
        return rom.get_object_by_id(obj_id)
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
    Get a ManagedRigidObject or ManagedArticulatedObject from its handle.

    :param sim: The Simulator instance.
    :param obj_handle: object istance handle for which ManagedObject is desired.

    :return: a ManagedObject or None
    """

    rom = sim.get_rigid_object_manager()
    if rom.get_library_has_handle(obj_handle):
        return rom.get_object_by_handle(obj_handle)
    aom = sim.get_articulated_object_manager()
    if aom.get_library_has_handle(obj_handle):
        return aom.get_object_by_handle(obj_handle)

    return None


def get_object_set_from_id_set(
    sim: habitat_sim.Simulator,
    id_set: List[int],
    ao_link_map: Optional[Dict[int, int]] = None,
) -> List[
    Union[
        habitat_sim.physics.ManagedRigidObject,
        habitat_sim.physics.ManagedArticulatedObject,
    ]
]:
    """
    Get the ManagedObjects from a set of object_ids.

    :param sim: The Simulator instance.
    :param id_set: The set of object ids for which ManagedObjects are desired.
    :param ao_link_map: A pre-computed map from link object ids to their parent ArticulatedObject's object id.

    :return: a list of tuples, first element is a ManagedObject, second is an optional link index.
    """

    if ao_link_map is None:
        # Note: better to pre-compute this and pass it around
        ao_link_map = get_ao_link_id_map(sim)

    rom = sim.get_rigid_object_manager()
    aom = sim.get_articulated_object_manager()
    rigids = [
        (rom.get_object_by_id(ro_id), None)
        for ro_id in id_set
        if rom.get_library_has_id(ro_id)
    ]
    aos = [
        (
            aom.get_object_by_id(ao_link_map[ao_id]),
            aom.get_object_by_id(ao_link_map[ao_id]).link_object_ids[ao_id],
        )
        for ao_id in id_set
        if ao_id in ao_link_map
    ]

    return rigids + aos


def get_obj_contact_pairs(
    sim: habitat_sim.Simulator,
    obj: Union[
        habitat_sim.physics.ManagedArticulatedObject,
        habitat_sim.physics.ManagedRigidObject,
    ],
    ao_link_map: Optional[Dict[int, int]] = None,
    do_collision_detection: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """
    Search contact points for this object and list any objects contacting this one with some details.

    :param sim: The Simulator instance.
    :param obj: The ManagedObject instance.
    :param ao_link_map: A pre-computed map from link object ids to their parent ArticulatedObject's object id.
    :param do_collision_detection: Whether or not to run discrete collision detection before querying contact points. Should be True if called in isolation and False if called as part of a larger state investigation after a sim step or recent collision detection call.

    :return: A dict mapping contacting object ids to contact detail summary.
    """

    if ao_link_map is None:
        ao_link_map = get_ao_link_id_map(sim)
    my_obj_id = obj.object_id

    def fill_defaults():
        return {"object_handle": "", "deepest_dist": 99999, "num_points": 0}

    if do_collision_detection:
        sim.perform_discrete_collision_detection()

    contacting_object_details = {}
    for cp in sim.get_physics_contact_points():
        contacting_obj_id = None
        if cp.object_id_a == my_obj_id:
            contacting_obj_id = cp.object_id_b
        if cp.object_id_b == my_obj_id:
            contacting_obj_id = cp.object_id_a
        if contacting_obj_id is not None:
            contacting_obj = get_object_set_from_id_set(
                sim, id_set=[contacting_obj_id], ao_link_map=ao_link_map
            )[0][0]
            if contacting_obj_id not in contacting_object_details:
                contacting_object_details[contacting_obj_id] = fill_defaults()
            contacting_object_details[contacting_obj_id][
                "object_handle"
            ] = contacting_obj.handle
            contacting_object_details[contacting_obj_id]["deepest_dist"] = min(
                contacting_object_details[contacting_obj_id]["deepest_dist"],
                cp.contact_distance,
            )
            contacting_object_details[contacting_obj_id]["num_points"] += 1

    print(contacting_object_details)
    return contacting_object_details

    # from habitat.sims.habitat_simulator.sim_utilities import get_obj_contact_pairs
    # get_obj_contact_pairs(self.sim,new_objs[0][0])


def above(
    sim: habitat_sim.Simulator,
    objectA: Union[
        habitat_sim.physics.ManagedRigidObject,
        habitat_sim.physics.ManagedArticulatedObject,
    ],
    ao_link_map: Optional[Dict[int, int]] = None,
) -> List[
    Tuple[
        Union[
            habitat_sim.physics.ManagedRigidObject,
            habitat_sim.physics.ManagedArticulatedObject,
        ],
        Optional[int],
    ]
]:
    """
    Get a list of all objects that a particular objectA is 'above'.
    Concretely, 'above' is defined as: a downward raycast of any object keypoint hits the object below.

    :param sim: The Simulator instance.
    :param objectA: The ManagedRigidObject for which to query the 'above' set.
    :param ao_link_map: A pre-computed map from link object ids to their parent ArticulatedObject's object id.

    :return: a list of tuples, first element is a ManagedObject, second is an optional link index.
    """

    # get object ids of all objects below this one
    above_object_ids = [
        hit.object_id
        for keypoint_raycast_result in object_keypoint_cast(sim, objectA)
        for hit in keypoint_raycast_result.hits
    ]
    above_object_ids = list(set(above_object_ids))

    above_objects_links = get_object_set_from_id_set(
        sim, above_object_ids, ao_link_map
    )

    # attempt to remove self from the list if present
    above_objects_links = [
        obj_link
        for obj_link in above_objects_links
        if obj_link[0] != objectA.object_id
    ]

    return above_objects_links


def below(
    sim: habitat_sim.Simulator,
    objectA: Union[
        habitat_sim.physics.ManagedRigidObject,
        habitat_sim.physics.ManagedArticulatedObject,
    ],
    ao_link_map: Optional[Dict[int, int]] = None,
) -> List[
    Tuple[
        Union[
            habitat_sim.physics.ManagedRigidObject,
            habitat_sim.physics.ManagedArticulatedObject,
        ],
        Optional[int],
    ]
]:
    """
    Get a list of all objects that a particular objectA is 'below'.
    Concretely, 'below' is defined as: an upward raycast of any object keypoint hits the object above.

    :param sim: The Simulator instance.
    :param objectA: The ManagedRigidObject for which to query the 'above' set.
    :param ao_link_map: A pre-computed map from link object ids to their parent ArticulatedObject's object id.

    :return: a list of tuples, first element is a ManagedObject, second is an optional link index.
    """

    # get object ids of all objects below this one
    below_object_ids = [
        hit.object_id
        for keypoint_raycast_result in object_keypoint_cast(
            sim, objectA, direction=mn.Vector3(0, 1, 0)
        )
        for hit in keypoint_raycast_result.hits
    ]
    below_object_ids = list(set(below_object_ids))

    below_objects_links = get_object_set_from_id_set(
        sim, below_object_ids, ao_link_map
    )

    # attempt to remove self from the list if present
    below_objects_links = [
        obj_link
        for obj_link in below_objects_links
        if obj_link[0] != objectA.object_id
    ]

    return below_objects_links


def within(
    sim: habitat_sim.Simulator,
    objectA: Union[
        habitat_sim.physics.ManagedRigidObject,
        habitat_sim.physics.ManagedArticulatedObject,
    ],
    ao_link_map: Optional[Dict[int, int]] = None,
) -> List[
    Tuple[
        Union[
            habitat_sim.physics.ManagedRigidObject,
            habitat_sim.physics.ManagedArticulatedObject,
        ],
        Optional[int],
    ]
]:
    """
    Get a list of all objects that a particular objectA is 'within'.
    Concretely, 'within' is defined as: a threshold number of opposing keypoing raycasts hit the same object.
    This function computes raycasts along all global axes from all keypoints and checks opposing rays for collision with the same object.

    :param sim: The Simulator instance.
    :param objectA: The ManagedRigidObject for which to query the 'within' set.
    :param ao_link_map: A pre-computed map from link object ids to their parent ArticulatedObject's object id.

    :return: a list of tuples, first element is a ManagedObject, second is an optional link index.
    """

    global_keypoints = get_object_global_keypoints(objectA)

    # build axes vectors
    pos_axes = [mn.Vector3.x_axis(), mn.Vector3.y_axis(), mn.Vector3.z_axis()]
    neg_axes = [-1 * axis for axis in pos_axes]

    # raycast for each axis for each keypoint
    keypoint_intersect_set: List[List[int]] = [
        [] for _ in range(len(global_keypoints))
    ]
    for k_ix, keypoint in enumerate(global_keypoints):
        for a_ix in range(3):
            [
                hit.object_id
                for keypoint_raycast_result in object_keypoint_cast(
                    sim, objectA
                )
                for hit in keypoint_raycast_result.hits
            ]
            pos_ids = [
                hit.object_id
                for hit in sim.cast_ray(
                    habitat_sim.geo.Ray(keypoint, pos_axes[a_ix]),
                    max_distance=1.0,
                ).hits
            ]
            neg_ids = [
                hit.object_id
                for hit in sim.cast_ray(
                    habitat_sim.geo.Ray(keypoint, neg_axes[a_ix]),
                    max_distance=1.0,
                ).hits
            ]
            intersect_ids = [obj_id for obj_id in pos_ids if obj_id in neg_ids]
            keypoint_intersect_set[k_ix].extend(intersect_ids)
        keypoint_intersect_set[k_ix] = list(set(keypoint_intersect_set[k_ix]))

    # initialize the list from keypoint 0 (center of mass) which gaurantees containment
    containment_ids = list(keypoint_intersect_set[0])
    # "vote" for ids from other keypoints
    id_votes: defaultdict[int, int] = defaultdict(lambda: 0)
    for k_ix in range(1, len(global_keypoints)):
        for k_ix_2 in range(1, len(global_keypoints)):
            if k_ix < k_ix_2:
                for obj_id in keypoint_intersect_set[k_ix]:
                    if obj_id in keypoint_intersect_set[k_ix_2]:
                        id_votes[obj_id] += 1

    # count votes for other keypoints and de-duplicate
    containment_ids = containment_ids + [
        obj_id for obj_id in id_votes if id_votes[obj_id] > 2
    ]
    containment_ids = list(set(containment_ids))

    within_objects_links = get_object_set_from_id_set(
        sim, containment_ids, ao_link_map
    )

    # attempt to remove self from the list if present
    within_objects_links = [
        obj_link
        for obj_link in within_objects_links
        if obj_link[0] != objectA.object_id
    ]

    return within_objects_links


def ontop(
    sim: habitat_sim.Simulator,
    objectA: Union[
        habitat_sim.physics.ManagedRigidObject,
        habitat_sim.physics.ManagedArticulatedObject,
    ],
    ao_link_map: Optional[Dict[int, int]] = None,
    do_collision_detection: bool = True,
) -> List[
    Union[
        habitat_sim.physics.ManagedRigidObject,
        habitat_sim.physics.ManagedArticulatedObject,
    ]
]:
    """
    Get a list of all objects that are "ontop" of a particular objectA.
    Concretely, 'ontop' is defined as: contact points between objectA and objectB have vertical normals "upward" relative to objectA.
    This function uses collision points to determine which objects are resting on or contacting the surface of objectA.

    :param sim: The Simulator instance.
    :param objectA: The ManagedRigidObject for which to query the 'ontop' set.
    :param ao_link_map: A pre-computed map from link object ids to their parent ArticulatedObject's object id.
    :param do_collision_detection: If True, a fresh discrete collision detection is run before the contact point query. Pass False to skip if a recent sim step or pre-process has run a collision detection pass on the current state.

    :return: a list of tuples, first element is a ManagedObject, second is an optional link index.
    """

    if ao_link_map is None:
        ao_link_map = get_ao_link_id_map(sim)

    if do_collision_detection:
        sim.perform_discrete_collision_detection()

    yup = mn.Vector3(0.0, 1.0, 0.0)
    up_threshold = 0.75

    ontop_objects = []
    for cp in sim.get_physics_contact_points():
        contacting_obj_id = None
        obj_is_b = False
        if cp.object_id_a == objectA.object_id:
            contacting_obj_id = cp.object_id_b
        if cp.object_id_b == objectA.object_id:
            contacting_obj_id = cp.object_id_a
            obj_is_b = True
        if contacting_obj_id is not None:
            contact_dir_me = (
                cp.contact_normal_on_b_in_ws
                if obj_is_b
                else -cp.contact_normal_on_b_in_ws
            )
            if mn.math.dot(contact_dir_me, yup) > up_threshold:
                contacting_obj = get_obj_from_id(
                    sim, contacting_obj_id, ao_link_map
                )
                ontop_objects.append(contacting_obj)

    return ontop_objects


def nearby(
    sim: habitat_sim.Simulator,
    objectA: Union[
        habitat_sim.physics.ManagedRigidObject,
        habitat_sim.physics.ManagedArticulatedObject,
    ],
    distance: float = 1.0,
    size_regularized: bool = True,
    geodesic: bool = False,
    alt_pathfinder=None,
    island_id: int = -1,
    ao_link_map: Optional[Dict[int, int]] = None,
) -> List[
    Union[
        habitat_sim.physics.ManagedRigidObject,
        habitat_sim.physics.ManagedArticulatedObject,
    ]
]:
    """
    Get list of ManagedRigidObjects and ManagedArticulatedObjects which are "nearby" the target object.
    Nearby is defined by L2 or geodesic distance (if geodesic==True) between center of mass points within "distance" threshold.
    #TODO: should this be keypoints instead of CoM?

    :param sim: The Simulator instance.
    :param objectA: The ManagedRigidObject for which to query the 'ontop' set.
    :param distance: Target threshold distance for "nearby". How close do objects need to be?
    :param size_regularized: If True, regularize distance by object size. Use bounding box to determine regularization weight.
    :param geodesic: If True, use navmesh distance instead of L2 distance. Counts distance to navmesh + distance on navmesh.
    :param alt_pathfinder: If geodesic, optionally provide an alternative pathfinder instance. If not provided, uses sim.pathfinder.
    :param island_id: If geodesic, restrict the check to a particular navmesh. Should be largest_navmesh_island. -1 is all islands, distance checks could be NaN. NaN distance counts as not nearby.
    :param ao_link_map: A pre-computed map from link object ids to their parent ArticulatedObject's object id.
    """

    all_objs = get_all_objects(sim)
    # first get L2 distances

    all_l2_nearby_objs = [
        obj
        for obj in all_objs
        if (obj.translation - objectA.translation).length() <= distance
    ]

    # my_size_buffer = 0
    # if size_regularized:
    #     # TODO: my_bb = get_bb_for_object_id()
    #     pass

    # for obj in all_objs:
    #     obj_size_buffer = 0
    #     if size_regularized:
    #         # TODO:
    #         pass

    if geodesic:
        all_geo_nearby: List[
            Union[
                habitat_sim.physics.ManagedRigidObject,
                habitat_sim.physics.ManagedArticulatedObject,
            ]
        ] = []
        # first setup for geodesic
        if alt_pathfinder is None:
            alt_pathfinder = sim.pathfinder
        assert alt_pathfinder.is_loaded
        my_snap = alt_pathfinder.snap_point(
            objectA.translation, island_index=island_id
        )
        my_snap_dist = (my_snap - objectA.translation).length()
        shortest_path = habitat_sim.nav.ShortestPath()
        shortest_path.requested_start = my_snap
        if my_snap_dist >= distance:
            # too far from navmesh, empty return
            return all_geo_nearby
        # now filter by geodesic if necessary
        for obj in all_l2_nearby_objs:
            obj_snap = alt_pathfinder.snap_point(
                obj.translation, island_index=island_id
            )
            obj_snap_dist = (obj_snap - obj.translation).length()
            # threshold for allowed navmesh travel
            min_geo = distance - obj_snap_dist - my_snap_dist
            if min_geo >= distance:
                continue
            shortest_path.requested_end = obj_snap
            found_path = alt_pathfinder.find_path(shortest_path)
            if not found_path:
                continue
            geo_dist = shortest_path.geodesic_distance
            if geo_dist < min_geo:
                # we found a "nearby" point
                all_geo_nearby.append(obj)
        return all_geo_nearby
    return all_l2_nearby_objs


# ============================================================
# Debug Rendering Utils (move to debug_visualizer.py?)
# ============================================================


def debug_draw_bb(
    sim: habitat_sim.Simulator,
    bb: mn.Range3D,
    transform: mn.Matrix4 = None,
    color: Optional[mn.Color4] = None,
) -> None:
    """
    Render the AABB with DebugLineRender utility at the current frame.
    Must be called after each frame is rendered, before querying the image data.

    :param sim: The Simulator instance.
    :param bb: The bounding box to render.
    :param transform: An optional local to global transform for moving the bounding box.
    :param color: An optional wireframe render color. Default to magenta.
    """

    # draw the box
    if color is None:
        color = mn.Color4.magenta()
    if transform is None:
        transform = mn.Matrix4()
    dblr = sim.get_debug_line_render()
    dblr.push_transform(transform)
    dblr.draw_box(bb.min, bb.max, color)
    dblr.pop_transform()


def debug_draw_rigid_object_bb(
    sim: habitat_sim.Simulator,
    objectA: habitat_sim.physics.ManagedRigidObject,
    color: Optional[mn.Color4] = None,
) -> None:
    """
    Render the AABB of an object with DebugLineRender utility at the current frame.
    Must be called after each frame is rendered, before querying the image data.

    :param sim: The Simulator instance.
    :param objectA: The ManagedRigidObject for which to render the bounding box.
    :param color: An optional wireframe render color. Default to magenta.
    """

    debug_draw_bb(
        sim,
        objectA.root_scene_node.cumulative_bb,
        objectA.transformation,
        color,
    )


def debug_draw_selected_set(
    sim: habitat_sim.Simulator,
    selected_set: List[
        Tuple[
            Union[
                habitat_sim.physics.ManagedRigidObject,
                habitat_sim.physics.ManagedArticulatedObject,
            ],
            Optional[int],
        ]
    ],
    ao_aabbs: Dict[int, mn.Range3D] = None,
) -> None:
    """
    Render the selected set (e.g. "above" set) for the selected object and draw a debug visualization.

    :param sim: The Simulator instance.
    :param selected_set: The set of selected objects for which to render bounding boxes. Consists of a list of Tuples, each with a ManagedObject and optional link index.
    """

    if ao_aabbs is None:
        ao_aabbs = get_ao_root_bbs(sim)
    rendered_base = []
    for set_obj, link_id in selected_set:
        if type(set_obj) == habitat_sim.physics.ManagedBulletRigidObject:
            debug_draw_rigid_object_bb(sim, set_obj, color=mn.Color4.green())
        else:
            if set_obj.object_id not in rendered_base:
                rendered_base.append(set_obj.object_id)
                debug_draw_bb(
                    sim,
                    ao_aabbs[set_obj.object_id],
                    set_obj.transformation,
                    color=mn.Color4.blue(),
                )
            if link_id not in set_obj.get_link_ids():
                raise AssertionError("Link id not found, should not get here.")
            link_node = set_obj.get_link_scene_node(link_id)
            link_transform = link_node.absolute_transformation()
            debug_draw_bb(
                sim, link_node.cumulative_bb, link_transform, mn.Color4.cyan()
            )
