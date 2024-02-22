#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

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


def get_rigid_object_global_keypoints(
    objectA: habitat_sim.physics.ManagedRigidObject,
) -> List[mn.Vector3]:
    """
    Get a list of rigid object keypoints in global space.
    0th point is the bounding box center, others are bounding box corners.

    :param objectA: The ManagedRigidObject from which to extract keypoints.

    :return: A set of global 3D keypoints for the object.
    """

    bb = objectA.root_scene_node.cumulative_bb
    local_keypoints = [bb.center()]
    local_keypoints.extend(get_bb_corners(bb))
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
    Computes object global keypoints, casts rays from each in the specified direction and returns the resulting RaycastResults.

    :param sim: The Simulator instance.
    :param objectA: The ManagedRigidObject from which to extract keypoints and raycast.
    :param direction: Optionally provide a unit length global direction vector for the raycast. If None, default to -Y.

    :return: A list of RaycastResults, one from each object keypoint.
    """

    if direction is None:
        # default to downward raycast
        direction = mn.Vector3(0, -1, 0)

    global_keypoints = get_rigid_object_global_keypoints(objectA)
    return [
        sim.cast_ray(habitat_sim.geo.Ray(keypoint, direction))
        for keypoint in global_keypoints
    ]


# ============================================================
# Utilities for Querying Object Relationships
# ============================================================


def above(
    sim: habitat_sim.Simulator,
    objectA: Union[
        habitat_sim.physics.ManagedRigidObject,
        habitat_sim.physics.ManagedArticulatedObject,
    ],
) -> List[int]:
    """
    Get a list of all objects that a particular objectA is 'above'.
    Concretely, 'above' is defined as: a downward raycast of any object keypoint hits the object below.

    :param sim: The Simulator instance.
    :param objectA: The ManagedRigidObject for which to query the 'above' set.

    :return: a list of object ids.
    """

    # get object ids of all objects below this one
    above_object_ids = [
        hit.object_id
        for keypoint_raycast_result in object_keypoint_cast(sim, objectA)
        for hit in keypoint_raycast_result.hits
    ]
    above_object_ids = list(set(above_object_ids))

    # remove self from the list if present
    if objectA.object_id in above_object_ids:
        above_object_ids.remove(objectA.object_id)

    return above_object_ids


def within(
    sim: habitat_sim.Simulator,
    objectA: Union[
        habitat_sim.physics.ManagedRigidObject,
        habitat_sim.physics.ManagedArticulatedObject,
    ],
    max_distance: float = 1.0,
    keypoint_vote_threshold: int = 2,
    center_ensures_containment: bool = True,
) -> List[int]:
    """
    Get a list of all objects that a particular objectA is 'within'.
    Concretely, 'within' is defined as: a threshold number of opposing keypoint raycasts hit the same object.
    This function computes raycasts along all global axes from all keypoints and checks opposing rays for collision with the same object.

    :param sim: The Simulator instance.
    :param objectA: The ManagedRigidObject for which to query the 'within' set.
    :param max_distance: The maximum ray distance to check in each opposing direction (this is half the "wingspan" of the check). Makes the raycast more efficienct and realistically containing objects will have a limited size.
    :param keypoint_vote_threshold: The minimum number of keypoints which must indicate containment to qualify objectA as "within" another object.
    :param center_ensures_containment: If True, positive test of objectA's center keypoint alone qualifies objectA as "within" another object.

    :return: a list of object_id integers.
    """

    global_keypoints = get_rigid_object_global_keypoints(objectA)

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
    if objectA.object_id in containment_ids:
        containment_ids.remove(objectA.object_id)

    return containment_ids
