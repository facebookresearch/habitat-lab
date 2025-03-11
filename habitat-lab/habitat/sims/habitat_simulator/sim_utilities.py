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
    dbv: Optional[DebugVisualizer] = None,
) -> bool:
    """
    Attempt to project an object in the gravity direction onto the surface below it.

    :param sim: The Simulator instance.
    :param obj: The RigidObject instance.
    :param support_obj_ids: A list of object ids designated as valid support surfaces for object placement. Contact with other objects is a criteria for placement rejection. If none provided, default support surface is the stage/ground mesh (0).
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
        support_obj_ids = [habitat_sim.stage_id]

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
                (cp.contact_distance < -0.05)
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


def get_ao_root_bb(
    ao: habitat_sim.physics.ManagedArticulatedObject,
) -> mn.Range3D:
    """
    Get the local bounding box of all links of an articulated object in the root frame.

    :param ao: The ArticulatedObject instance.
    """

    # NOTE: we'd like to use SceneNode AABB, but this won't work because the links are not in the subtree of the root:
    # ao.root_scene_node.compute_cumulative_bb()

    ao_local_part_bb_corners = []

    link_nodes = [ao.get_link_scene_node(ix) for ix in range(-1, ao.num_links)]
    for link_node in link_nodes:
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
    Get a ManagedRigidObject or ManagedArticulatedObject from its instance handle.

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


def get_rigid_object_global_keypoints(
    object_a: habitat_sim.physics.ManagedRigidObject,
) -> List[mn.Vector3]:
    """
    Get a list of rigid object keypoints in global space.
    0th point is the bounding box center, others are bounding box corners.

    :param object_a: The ManagedRigidObject from which to extract keypoints.

    :return: A set of global 3D keypoints for the object.
    """

    bb = object_a.root_scene_node.cumulative_bb
    return get_global_keypoints_from_bb(bb, object_a.transformation)


def get_articulated_object_global_keypoints(
    object_a: habitat_sim.physics.ManagedArticulatedObject,
    ao_aabbs: Dict[int, mn.Range3D] = None,
) -> List[mn.Vector3]:
    """
    Get global bb keypoints for an ArticulatedObject.

    :param object_a: The ManagedArticulatedObject from which to extract keypoints.
    :param ao_aabbs: A pre-computed map from ArticulatedObject object_ids to their local bounding boxes. If not provided, recomputed as necessary. Must contain the subjects of the query.

    :return: A set of global 3D keypoints for the object.
    """

    ao_bb = None
    if ao_aabbs is None:
        ao_bb = get_ao_root_bb(object_a)
    else:
        ao_bb = ao_aabbs[object_a.object_id]

    return get_global_keypoints_from_bb(ao_bb, object_a.transformation)


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
    ao_aabbs: Dict[int, mn.Range3D] = None,
) -> List[mn.Vector3]:
    """
    Get a list of object keypoints in global space given an object id.
    0th point is the center of bb, others are bounding box corners.

    :param sim: The Simulator instance.
    :param object_id: The integer id for the object from which to extract keypoints.
    :param ao_link_map: A pre-computed map from link object ids to their parent ArticulatedObject's object id. If not provided, recomputed as necessary.
    :param ao_aabbs: A pre-computed map from ArticulatedObject object_ids to their local bounding boxes. If not provided, recomputed as necessary.

    :return: A set of global 3D keypoints for the object.
    """

    obj = get_obj_from_id(sim, object_id, ao_link_map)

    if isinstance(obj, habitat_sim.physics.ManagedBulletRigidObject):
        return get_rigid_object_global_keypoints(obj)
    elif obj.object_id != object_id:
        # this is an ArticulatedLink
        return get_articulated_link_global_keypoints(
            obj, obj.link_object_ids[object_id]
        )
    else:
        # ArticulatedObject
        return get_articulated_object_global_keypoints(obj, ao_aabbs)


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

    global_keypoints = get_rigid_object_global_keypoints(object_a)
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

    global_keypoints = get_rigid_object_global_keypoints(object_a)

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
        # initialize the list from keypoint 0 (center of bounding box) which gaurantees containment
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
    Concretely, 'ontop' is defined as: contact points between object_a and objectB have vertical normals "upward" relative to object_a.
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
    ao_aabbs: Dict[int, mn.Range3D] = None,
) -> Tuple[bool, float]:
    """
    Check if an object is within a region by checking region containment of keypoints.

    :param sim: The Simulator instance.
    :param object_a: The object instance.
    :param region: The SemanticRegion to check.
    :param containment_threshold: threshold ratio of keypoints which need to be in a region to count as containment.
    :param center_only: If True, only use the BB center keypoint, all or nothing.
    :param ao_link_map: A pre-computed map from link object ids to their parent ArticulatedObject's object id.
    :param ao_aabbs: A pre-computed map from ArticulatedObject object_ids to their local bounding boxes. If not provided, recomputed as necessary.


    :return: boolean containment and the ratio of keypoints which are inside the region.
    """

    key_points = get_global_keypoints_from_object_id(
        sim,
        object_id=object_a.object_id,
        ao_link_map=ao_link_map,
        ao_aabbs=ao_aabbs,
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
    ao_aabbs: Dict[int, mn.Range3D] = None,
) -> List[Tuple[int, float]]:
    """
    Get a sorted list of regions containing an object using bounding box keypoints.

    :param sim: The Simulator instance.
    :param object_a: The object instance.
    :param ao_link_map: A pre-computed map from link object ids to their parent ArticulatedObject's object id.
    :param ao_aabbs: A pre-computed map from ArticulatedObject object_ids to their local bounding boxes. If not provided, recomputed as necessary.

    :return: A sorted list of region index, ratio pairs. First item in the list the primary containing region.
    """

    key_points = get_global_keypoints_from_object_id(
        sim,
        object_id=object_a.object_id,
        ao_link_map=ao_link_map,
        ao_aabbs=ao_aabbs,
    )

    return sim.semantic_scene.get_regions_for_points(key_points)
