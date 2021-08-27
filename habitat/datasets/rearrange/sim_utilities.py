#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, List, Any, Dict
import os.path as osp

import magnum as mn
import numpy as np

import habitat_sim


class DebugVisualizer(object):
    """
    Support class for simple visual debugging of a utility simulator.
    Assumes the default agent (0) is a camera (i.e. there exists an RGB sensor coincident with agent 0 transformation).
    """

    def __init__(
        self,
        sim: habitat_sim.Simulator,
        output_path: str ="visual_debug_output/",
        default_sensor_uuid: str ="rgb",
    )-> None:
        """
        Initialize the debugger provided a Simulator and the uuid of the debug sensor.
        NOTE: Expects the debug sensor attached to and coincident with agent 0's frame.
        """
        self.sim = sim
        self.output_path = output_path
        self.default_sensor_uuid = default_sensor_uuid
        self._debug_obs:List[Any] = []
        # NOTE: visualizations from the DebugLinerRender utility will only be visible in PINHOLE RGB sensor views
        self.debug_line_render = sim.get_debug_line_render()

    def look_at(self, look_at:mn.Vector3, look_from:Optional[mn.Vector3]=None, look_up:mn.Vector3=np.array([0, 1.0, 0]))-> None:
        """
        Point the debug camera at a target.
        """
        agent = self.sim.get_agent(0)
        camera_pos = (
            look_from
            if look_from is not None
            else agent.scene_node.translation
        )
        agent.scene_node.rotation = mn.Quaternion.from_matrix(
            mn.Matrix4.look_at(camera_pos, look_at, look_up).rotation()
        )

    def get_observation(self, look_at:Optional[mn.Vector3]=None, look_from:Optional[mn.Vector3]=None, obs_cache:Optional[List[Any]]=None)-> None:
        """
        Render a debug observation of the current state and cache it.
        Optionally configure the camera transform.
        Optionally provide an alternative observation cache.
        """
        if look_at is not None:
            self.look_at(look_at, look_from)
        if obs_cache is None:
            self._debug_obs.append(self.sim.get_sensor_observations())
        else:
            obs_cache.append(self.sim.get_sensor_observations())

    def make_debug_video(
        self, output_path:Optional[str]=None, prefix:str="", fps:int=4, obs_cache:Optional[List[Any]]=None
    )-> None:
        """
        Produce a video from a set of debug observations.
        """
        if output_path is None:
            output_path = self.output_path

        # if output directory doesn't exist, create it
        if not osp.exists(output_path):
            import os

            try:
                os.makedirs(output_path)
            except OSError:
                print(
                    f"DebugVisualizer: Aborting 'make_debug_video': Failed to create the output directory: {output_path}"
                )
                return
            print(
                f"DebugVisualizer: 'make_debug_video' output directory did not exist and was created: {output_path}"
            )

        # get a timestamp tag with current date and time for video name
        from datetime import datetime

        date_time = datetime.now().strftime("%m_%d_%Y_%H%M%S")

        if obs_cache is None:
            obs_cache = self._debug_obs

        from habitat_sim.utils import viz_utils as vut

        file_path = output_path + prefix + date_time
        print(f"DebugVisualizer: Saving debug video to {file_path}")
        vut.make_video(
            obs_cache, self.default_sensor_uuid, "color", file_path, fps=fps
        )


class Receptacle(object):
    """
    Stores parameters necessary to define a AABB Receptacle for object sampling.
    """

    def __init__(
        self,
        name:str,
        bounds: mn.Range3D,
        parent_object_handle: str=None,
        is_parent_object_articulated: bool=False,
        parent_link: int=-1, #-1 is base
    ) -> None:
        self.name = name
        self.bounds = bounds
        self.parent_object_handle = parent_object_handle
        self.is_parent_object_articulated = is_parent_object_articulated
        self.parent_link = parent_link

    def sample_uniform_local(self)-> mn.Vector3:
        """
        Sample a uniform random point in the local AABB.
        """
        return np.random.uniform(self.bounds.min, self.bounds.max)

    def sample_uniform_global(self, sim:habitat_sim.Simulator)-> mn.Vector3:
        """
        Sample a uniform random point in the local AABB and then transform it into global space.
        """
        local_sample = self.sample_uniform_local()
        if not self.is_parent_object_articulated:
            obj_mgr = sim.get_rigid_object_manager()
            obj = obj_mgr.get_object_by_handle(self.parent_object_handle)
            # NOTE: we use absolute transformation from the 2nd visual node (scaling node) and root of all render assets to correctly account for any COM shifting, re-orienting, or scaling which has been applied.
            return (
                obj.visual_scene_nodes[1]
                .absolute_transformation()
                .transform_point(local_sample)
            )
        else:
            ao_mgr = sim.get_articulated_object_manager()
            obj = ao_mgr.get_object_by_handle(self.parent_object_handle)
            # TODO: AO link in inertial frame?
            return (
                obj.get_link_scene_node(self.parent_link)
                .absolute_transformation()
                .transform_point(local_sample)
            )


def find_receptacles(sim)-> List[Receptacle]:
    """
    Return a list of all receptacles scraped from the scene's currently instanced objects.
    """
    #TODO: Receptacles should be screened if the orientation will not support placement.
    obj_mgr = sim.get_rigid_object_manager()
    ao_mgr = sim.get_articulated_object_manager()

    receptacles:List[Receptacle] = []

    # rigid objects
    for obj_handle in obj_mgr.get_object_handles():
        obj = obj_mgr.get_object_by_handle(obj_handle)
        user_attr = obj.user_attributes

        for sub_config_key in user_attr.get_subconfig_keys():
            if sub_config_key.startswith("bb_"):
                sub_config = user_attr.get_subconfig(sub_config_key)
                # this is a receptacle, parse it
                assert sub_config.has_value("position")
                assert sub_config.has_value("scale")
                receptacle_name = (
                    sub_config.get("name")
                    if sub_config.has_value("name")
                    else sub_config_key
                )
                receptacles.append(
                    Receptacle(
                        receptacle_name,
                        mn.Range3D.from_center(
                            sub_config.get("position"),
                            sub_config.get("scale"),
                        ),
                        obj_handle,
                    )
                )

    # articulated objects #TODO: merge with above
    for obj_handle in ao_mgr.get_object_handles():
        obj = ao_mgr.get_object_by_handle(obj_handle)
        user_attr = obj.user_attributes

        for sub_config_key in user_attr.get_subconfig_keys():
            if sub_config_key.startswith("bb_"):
                sub_config = user_attr.get_subconfig(sub_config_key)
                # this is a receptacle, parse it
                assert sub_config.has_value("position")
                assert sub_config.has_value("scale")
                assert sub_config.has_value("parent_link")
                receptacle_name = (
                    sub_config.get("name")
                    if sub_config.has_value("name")
                    else sub_config_key
                )
                parent_link_name = sub_config.get("parent_link")
                parent_link_ix = None
                for link in range(obj.num_links):
                    if obj.get_link_name(link) == parent_link_name:
                        parent_link_ix = link
                        break
                assert (
                    parent_link_ix is not None
                ), f"('parent_link' = '{parent_link_name}') in receptacle configuration does not match any model links."
                receptacles.append(
                    Receptacle(
                        receptacle_name,
                        mn.Range3D.from_center(
                            sub_config.get("position"),
                            sub_config.get("scale"),
                        ),
                        obj_handle,
                        is_parent_object_articulated=True,
                        parent_link=parent_link_ix,
                    )
                )

    return receptacles


def register_custom_wireframe_box_template(
    sim:habitat_sim.Simulator, size:mn.Vector3, template_name:str="custom_wireframe_box"
)-> str:
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
    sim:habitat_sim.Simulator, size:mn.Vector3, center:mn.Vector3, attach_to:Optional[habitat_sim.scene.SceneNode]=None, orientation:mn.Quaternion=mn.Quaternion()
)-> habitat_sim.physics.ManagedRigidObject:
    """
    Generate a wire box object and optionally attach it to another existing object (automatically applies object scale).
    Returns the new object.
    """
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


def add_transformed_wire_box(sim:habitat_sim.Simulator, size:mn.Vector3, transform:mn.Matrix4=mn.Matrix4())-> habitat_sim.physics.ManagedRigidObject:
    """
    Generate a transformed wire box in world space.
    Returns the new object.
    """
    box_template_handle = register_custom_wireframe_box_template(sim, size)
    new_object = sim.get_rigid_object_manager().add_object_by_template_handle(
        box_template_handle
    )
    new_object.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    new_object.collidable = False
    # translate to local offset if attached or global offset if not
    new_object.transformation = transform
    return new_object


def add_viz_sphere(sim:habitat_sim.Simulator, radius:float, pos:mn.Vector3)-> habitat_sim.physics.ManagedRigidObject:
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


def get_bb_corners(obj:habitat_sim.physics.ManagedRigidObject)-> List[mn.Vector3]:
    """
    Return a list of object bb corners in object local space.
    """
    bb = obj.root_scene_node.cumulative_bb
    return [
        bb.back_bottom_left,
        bb.back_bottom_right,
        bb.back_top_right,
        bb.back_top_left,
        bb.front_top_left,
        bb.front_top_right,
        bb.front_bottom_right,
        bb.front_bottom_left,
    ]


def bb_ray_prescreen(sim:habitat_sim.Simulator, obj:habitat_sim.physics.ManagedRigidObject, support_obj:habitat_sim.physics.ManagedRigidObject)-> Dict[str, Any]:
    """
    Pre-screen a potential placement by casting rays in gravity direction from each bb corner checking for interferring objects below.
    """
    lowest_key_point: mn.Vector3 = None
    lowest_key_point_height = None
    highest_support_impact: mn.Vector3 = None
    highest_support_impact_height = None
    raycast_results = []
    # we'll count rays which hit objects (obstructions) or the ground before the support surface
    obstructed = []
    grounded = []
    sup_obj_id = -1  # default ground if None
    if support_obj is not None:
        sup_obj_id = support_obj.object_id
    gravity_dir = sim.get_gravity().normalized()
    object_local_to_global = obj.transformation
    bb_corners = get_bb_corners(obj)
    key_points = [mn.Vector3(0)] + bb_corners  # [COM, c0, c1 ...]
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
        ray = habitat_sim.geo.Ray(world_point, gravity_dir)
        raycast_results.append(sim.cast_ray(ray))
        # classify any obstructions before hitting the support surface
        for hit in raycast_results[-1].hits:
            if hit.object_id == obj.object_id:
                continue
            elif hit.object_id == sup_obj_id:
                hit_point = ray.origin + ray.direction * hit.ray_distance
                support_impact_height = hit_point.projected_onto_normalized(
                    -gravity_dir
                )
                if (
                    highest_support_impact is None
                    or highest_support_impact_height < support_impact_height
                ):
                    highest_support_impact = hit_point
                    highest_support_impact_height = support_impact_height
            elif hit.object_id == -1:
                grounded.append(ix)
            else:
                obstructed.append(ix)
            break
    # compute the relative base height of the object from its lowest bb corner and COM
    base_rel_height = (
        lowest_key_point_height
        - obj.translation.projected_onto_normalized(-gravity_dir).length()
    )
    surface_snap_point = (
        None
        if highest_support_impact is None
        else highest_support_impact + gravity_dir * base_rel_height
    )
    # return list of obstructed and grounded rays, relative base height, distance to first surface impact, and ray results details
    return {
        "obstructed": obstructed,
        "grounded": grounded,
        "base_rel_height": base_rel_height,
        "surface_snap_point": surface_snap_point,
        "raycast_results": raycast_results,
    }


def snap_down(sim:habitat_sim.Simulator, obj:habitat_sim.physics.ManagedRigidObject, support_obj:Optional[habitat_sim.physics.ManagedRigidObject]=None, vdb:Optional[DebugVisualizer]=None)-> bool:
    """
    Project an object in gravity direction onto the surface below it and then correct for penetration given a target supporting surface or the ground.
    Optionally provide a DebugVisualizer (vdb)
    Returns boolean success. If successful, the object state is updated to the snapped location.
    """
    print("----------------------------")
    print("Snap-down")
    cached_position = obj.translation

    bb_ray_prescreen_results = bb_ray_prescreen(sim, obj, support_obj)

    if len(bb_ray_prescreen_results["obstructed"]) > 0:
        # prescreen ray hit a non-support object first, reject
        print(" Failure: obstruction (prescreen).")
        print("----------------------------")
        return False

    if bb_ray_prescreen_results["surface_snap_point"] is None:
        # no support under this object, return failure
        print(" Failure: no support hit (prescreen).")
        print("----------------------------")
        return False

    # TODO: perhaps we screen any ground hits to increase stability?
    if len(bb_ray_prescreen_results["grounded"]) > 1:
        # more than 1 prescreen ray(s) hit the ground, so probably not a stable placement
        print(
            " Failure: prescreen detected potentially unstable placement (hanging over ground)."
        )
        print("----------------------------")
        return False

    obj.translation = bb_ray_prescreen_results["surface_snap_point"]
    best_valid_position = obj.translation

    # finish up
    if best_valid_position is not None:
        # accept the final location if a valid location exists
        print(" Success: valid position found.")
        print("-----------------")
        obj.translation = best_valid_position
        if vdb is not None:
            vdb.get_observation(obj.translation)
        sim.perform_discrete_collision_detection()
        cps = sim.get_physics_contact_points()
        for cp in cps:
            if (
                cp.object_id_a == obj.object_id
                or cp.object_id_b == obj.object_id
            ):
                if cp.contact_distance < -0.01:
                    obj.translation = cached_position
                    print(" Failure: contact in final position.")
                    print("-----------------")
                    return False
                elif support_obj is not None:
                    if not (
                        cp.object_id_a == support_obj.object_id
                        or cp.object_id_b == support_obj.object_id
                    ):
                        obj.translation = cached_position
                        print(" Failure: contact in final position.")
                        print("-----------------")
                        return False
        return True
    else:
        # no valid position found, reset and return failure
        obj.translation = cached_position
        print(" Failure: no valid position found.")
        print("-----------------")
        return False
