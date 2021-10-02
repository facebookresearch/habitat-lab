#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os.path as osp
from typing import Any, Dict, List, Optional, Union

import magnum as mn
import numpy as np

import habitat_sim


class DebugVisualizer:
    """
    Support class for simple visual debugging of a utility simulator.
    Assumes the default agent (0) is a camera (i.e. there exists an RGB sensor coincident with agent 0 transformation).
    """

    def __init__(
        self,
        sim: habitat_sim.Simulator,
        output_path: str = "visual_debug_output/",
        default_sensor_uuid: str = "rgb",
    ) -> None:
        """
        Initialize the debugger provided a Simulator and the uuid of the debug sensor.
        NOTE: Expects the debug sensor attached to and coincident with agent 0's frame.
        """
        self.sim = sim
        self.output_path = output_path
        self.default_sensor_uuid = default_sensor_uuid
        self._debug_obs: List[Any] = []
        # NOTE: visualizations from the DebugLinerRender utility will only be visible in PINHOLE RGB sensor views
        self.debug_line_render = sim.get_debug_line_render()

    def look_at(
        self,
        look_at: mn.Vector3,
        look_from: Optional[mn.Vector3] = None,
        look_up: Optional[mn.Vector3] = None,
    ) -> None:
        """
        Point the debug camera at a target.
        """
        agent = self.sim.get_agent(0)
        camera_pos = (
            look_from
            if look_from is not None
            else agent.scene_node.translation
        )
        if look_up is None:
            # pick a valid "up" vector.
            look_dir = look_at - camera_pos
            look_up = (
                mn.Vector3(0, 1.0, 0)
                if look_dir[0] != 0 or look_dir[2] != 0
                else mn.Vector3(1.0, 0, 0)
            )
        agent.scene_node.rotation = mn.Quaternion.from_matrix(
            mn.Matrix4.look_at(camera_pos, look_at, look_up).rotation()
        )
        agent.scene_node.translation = camera_pos

    def get_observation(
        self,
        look_at: Optional[mn.Vector3] = None,
        look_from: Optional[mn.Vector3] = None,
        obs_cache: Optional[List[Any]] = None,
    ) -> None:
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

    def save_observation(
        self,
        output_path: Optional[str] = None,
        prefix: str = "",
        look_at: Optional[mn.Vector3] = None,
        look_from: Optional[mn.Vector3] = None,
        obs_cache: Optional[List[Any]] = None,
        show: bool = True,
    ) -> str:
        """
        Get an observation and save it to file.
        Return the filepath.
        """
        obs_cache = []
        self.get_observation(look_at, look_from, obs_cache)
        # save the obs as an image
        if output_path is None:
            output_path = self.output_path
        check_make_dir(output_path)
        from habitat_sim.utils import viz_utils as vut

        image = vut.observation_to_image(obs_cache[0]["rgb"], "color")
        from datetime import datetime

        # filename format "prefixmonth_day_year_hourminutesecondmicrosecond.png"
        date_time = datetime.now().strftime("%m_%d_%Y_%H%M%S%f")
        file_path = output_path + prefix + date_time + ".png"
        image.save(file_path)
        if show:
            image.show()
        return file_path

    def peek_rigid_object(
        self,
        obj: habitat_sim.physics.ManagedRigidObject,
        cam_local_pos: Optional[mn.Vector3] = None,
        peek_all_axis: bool = False,
    ) -> str:
        """
        Specialization to peek a rigid object.
        See _peek_object.
        """

        return self._peek_object(
            obj,
            obj.root_scene_node.cumulative_bb,
            cam_local_pos,
            peek_all_axis,
        )

    def peek_articulated_object(
        self,
        obj: habitat_sim.physics.ManagedArticulatedObject,
        cam_local_pos: Optional[mn.Vector3] = None,
        peek_all_axis: bool = False,
    ) -> str:
        """
        Specialization to peek an articulated object.
        See _peek_object.
        """

        obj_bb = get_ao_global_bb(obj)

        return self._peek_object(obj, obj_bb, cam_local_pos, peek_all_axis)

    def _peek_object(
        self,
        obj: Union[
            habitat_sim.physics.ManagedArticulatedObject,
            habitat_sim.physics.ManagedRigidObject,
        ],
        obj_bb: mn.Range3D,
        cam_local_pos: Optional[mn.Vector3] = None,
        peek_all_axis: bool = False,
    ) -> str:
        """
        Compute a camera placement to view an ArticulatedObject and show/save an observation.
        Return the filepath.
        If peek_all_axis, then create a merged 3x2 matrix of images looking at the object from all angles.
        """
        obj_abs_transform = obj.root_scene_node.absolute_transformation()
        look_at = obj_abs_transform.translation
        bb_size = obj_bb.size()
        # TODO: query fov and aspect from the camera spec
        fov = 90
        aspect = 0.75
        import math

        # compute the optimal view distance from the camera specs and object size
        distance = (np.amax(np.array(bb_size)) / aspect) / math.tan(
            fov / (360 / math.pi)
        )
        if cam_local_pos is None:
            # default to -Z (forward) of the object
            cam_local_pos = mn.Vector3(0, 0, -1)
        if not peek_all_axis:
            look_from = (
                obj_abs_transform.transform_vector(cam_local_pos).normalized()
                * distance
                + look_at
            )
            return self.save_observation(
                prefix="peek_" + obj.handle,
                look_at=look_at,
                look_from=look_from,
            )
        else:
            # collect axis observations
            axis_obs: List[Any] = []
            for axis in range(6):
                axis_vec = mn.Vector3()
                axis_vec[axis % 3] = 1 if axis // 3 == 0 else -1
                look_from = (
                    obj_abs_transform.transform_vector(axis_vec).normalized()
                    * distance
                    + look_at
                )
                self.get_observation(look_at, look_from, axis_obs)
            # stitch images together
            stitched_image = None
            from PIL import Image

            from habitat_sim.utils import viz_utils as vut

            for ix, obs in enumerate(axis_obs):
                image = vut.observation_to_image(obs["rgb"], "color")
                if stitched_image is None:
                    stitched_image = Image.new(
                        image.mode, (image.size[0] * 3, image.size[1] * 2)
                    )
                location = (
                    image.size[0] * (ix % 3),
                    image.size[1] * (0 if ix // 3 == 0 else 1),
                )
                stitched_image.paste(image, location)
            stitched_image.show()
        return ""

    def make_debug_video(
        self,
        output_path: Optional[str] = None,
        prefix: str = "",
        fps: int = 4,
        obs_cache: Optional[List[Any]] = None,
    ) -> None:
        """
        Produce a video from a set of debug observations.
        """
        if output_path is None:
            output_path = self.output_path

        check_make_dir(output_path)

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


def check_make_dir(output_path: str) -> None:
    """
    Check for the existence of the provided path and create it if not found.
    """
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


class Receptacle:
    """
    Stores parameters necessary to define a AABB Receptacle for object sampling.
    """

    def __init__(
        self,
        name: str,
        bounds: mn.Range3D,
        up: Optional[
            mn.Vector3
        ] = None,  # used for culling, optional in config
        parent_object_handle: str = None,
        is_parent_object_articulated: bool = False,
        parent_link: int = -1,  # -1 is base
    ) -> None:
        self.name = name
        self.bounds = bounds
        self.up = (
            up if up is not None else mn.Vector3.y_axis(1.0)
        )  # default local Y up
        self.parent_object_handle = parent_object_handle
        self.is_parent_object_articulated = is_parent_object_articulated
        self.parent_link = parent_link

    def sample_uniform_local(self) -> mn.Vector3:
        """
        Sample a uniform random point in the local AABB.
        """
        return np.random.uniform(self.bounds.min, self.bounds.max)

    def get_global_transform(self, sim: habitat_sim.Simulator) -> mn.Matrix4:
        """
        Isolates boilerplate necessary to extract receptacle global transform from ROs and AOs.
        """
        if not self.is_parent_object_articulated:
            obj_mgr = sim.get_rigid_object_manager()
            obj = obj_mgr.get_object_by_handle(self.parent_object_handle)
            # NOTE: we use absolute transformation from the 2nd visual node (scaling node) and root of all render assets to correctly account for any COM shifting, re-orienting, or scaling which has been applied.
            return obj.visual_scene_nodes[1].absolute_transformation()
        else:
            ao_mgr = sim.get_articulated_object_manager()
            obj = ao_mgr.get_object_by_handle(self.parent_object_handle)
            return obj.get_link_scene_node(
                self.parent_link
            ).absolute_transformation()

    def sample_uniform_global(self, sim: habitat_sim.Simulator) -> mn.Vector3:
        """
        Sample a uniform random point in the local AABB and then transform it into global space.
        """
        local_sample = self.sample_uniform_local()
        return self.get_global_transform(sim).transform_point(local_sample)


def get_all_scenedataset_receptacles(sim) -> Dict[str, Dict[str, List[str]]]:
    """
    Scrapes the active SceneDataset from a Simulator for all receptacles defined in rigid and articulated object templates.
    TODO: Note this will not include scene-specific overwrites, only receptacles included in object_config.json and ao_config.json files.
    """
    # cache the rigid and articulated receptacles seperately
    receptacles: Dict[str, Dict[str, List[str]]] = {
        "rigid": {},
        "articulated": {},
    }

    # first scrape the rigid object configs:
    rotm = sim.get_object_template_manager()
    for template_handle in rotm.get_template_handles(""):
        obj_template = rotm.get_template_by_handle(template_handle)
        for item in obj_template.get_user_config().get_subconfig_keys():
            if item.startswith("receptacle_"):
                if template_handle not in receptacles["rigid"]:
                    receptacles["rigid"][template_handle] = []
                receptacles["rigid"][template_handle].append(item)

    # TODO: we currently need to load every URDF to get at the configs. This should change once AO templates are better managed.
    aom = sim.get_articulated_object_manager()
    for urdf_handle, urdf_path in sim.metadata_mediator.urdf_paths.items():
        ao = aom.add_articulated_object_from_urdf(urdf_path)
        for item in ao.user_attributes.get_subconfig_keys():
            if item.startswith("receptacle_"):
                if urdf_handle not in receptacles["articulated"]:
                    receptacles["articulated"][urdf_handle] = []
                receptacles["articulated"][urdf_handle].append(item)
        aom.remove_object_by_handle(ao.handle)

    return receptacles


def find_receptacles(sim) -> List[Receptacle]:
    """
    Return a list of all receptacles scraped from the scene's currently instanced objects.
    """
    # TODO: Receptacles should be screened if the orientation will not support placement.
    obj_mgr = sim.get_rigid_object_manager()
    ao_mgr = sim.get_articulated_object_manager()

    receptacles: List[Receptacle] = []

    # rigid objects
    for obj_handle in obj_mgr.get_object_handles():
        obj = obj_mgr.get_object_by_handle(obj_handle)
        user_attr = obj.user_attributes

        for sub_config_key in user_attr.get_subconfig_keys():
            if sub_config_key.startswith("receptacle_"):
                sub_config = user_attr.get_subconfig(sub_config_key)
                # this is a receptacle, parse it
                assert sub_config.has_value("position")
                assert sub_config.has_value("scale")
                up = (
                    None
                    if not sub_config.has_value("up")
                    else sub_config.get("up")
                )

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
                        up,
                        obj_handle,
                    )
                )

    # articulated objects #TODO: merge with above
    for obj_handle in ao_mgr.get_object_handles():
        obj = ao_mgr.get_object_by_handle(obj_handle)
        user_attr = obj.user_attributes

        for sub_config_key in user_attr.get_subconfig_keys():
            if sub_config_key.startswith("receptacle_"):
                sub_config = user_attr.get_subconfig(sub_config_key)
                # this is a receptacle, parse it
                assert sub_config.has_value("position")
                assert sub_config.has_value("scale")
                up = (
                    None
                    if not sub_config.has_value("up")
                    else sub_config.get("up")
                )
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
                            sub_config.get("position") * obj.global_scale,
                            sub_config.get("scale") * obj.global_scale,
                        ),
                        up,
                        obj_handle,
                        is_parent_object_articulated=True,
                        parent_link=parent_link_ix,
                    )
                )

    return receptacles


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


def get_bb_corners(
    obj: habitat_sim.physics.ManagedRigidObject,
) -> List[mn.Vector3]:
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


def get_ao_global_bb(
    obj: habitat_sim.physics.ManagedArticulatedObject,
) -> mn.Range3D:
    """
    Compute the cumulative bb of an ao by merging all link bbs.
    """
    cumulative_global_bb = mn.Range3D()
    for link_ix in range(-1, obj.num_links):
        link_node = obj.get_link_scene_node(link_ix)
        bb = link_node.cumulative_bb
        global_bb = habitat_sim.geo.get_transformed_bb(
            bb, link_node.transformation
        )
        cumulative_global_bb = mn.math.join(cumulative_global_bb, global_bb)
    return cumulative_global_bb


def bb_ray_prescreen(
    sim: habitat_sim.Simulator,
    obj: habitat_sim.physics.ManagedRigidObject,
    support_obj_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Pre-screen a potential placement by casting rays in gravity direction from each bb corner checking for interferring objects below.
    """
    if support_obj_ids is None:
        support_obj_ids = [-1]  # default ground
    lowest_key_point: mn.Vector3 = None
    lowest_key_point_height = None
    highest_support_impact: mn.Vector3 = None
    highest_support_impact_height = None
    raycast_results = []
    gravity_dir = sim.get_gravity().normalized()
    object_local_to_global = obj.transformation
    bb_corners = get_bb_corners(obj)
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
        if ix == 0:
            ray = habitat_sim.geo.Ray(world_point, gravity_dir)
            raycast_results.append(sim.cast_ray(ray))
            # classify any obstructions before hitting the support surface
            for hit in raycast_results[-1].hits:
                if hit.object_id == obj.object_id:
                    continue
                elif hit.object_id in support_obj_ids:
                    hit_point = ray.origin + ray.direction * hit.ray_distance
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
                # terminates at the first non-self ray hit
                break
    # compute the relative base height of the object from its lowest bb corner and COM
    base_rel_height = (
        lowest_key_point_height
        - obj.translation.projected_onto_normalized(-gravity_dir).length()
    )

    surface_snap_point = (
        None
        if 0 not in support_impacts
        else support_impacts[0] + gravity_dir * base_rel_height
    )
    # return list of obstructed and grounded rays, relative base height, distance to first surface impact, and ray results details
    return {
        "base_rel_height": base_rel_height,
        "surface_snap_point": surface_snap_point,
        "raycast_results": raycast_results,
    }


def snap_down(
    sim: habitat_sim.Simulator,
    obj: habitat_sim.physics.ManagedRigidObject,
    support_obj_ids: Optional[List[int]] = None,
    vdb: Optional[DebugVisualizer] = None,
) -> bool:
    """
    Project an object in gravity direction onto the surface below it and then correct for penetration given a target supporting surface or the ground.
    Optionally provide a DebugVisualizer (vdb)
    Returns boolean success. If successful, the object state is updated to the snapped location.
    """
    cached_position = obj.translation

    if support_obj_ids is None:
        support_obj_ids = [-1]  # default ground

    bb_ray_prescreen_results = bb_ray_prescreen(sim, obj, support_obj_ids)

    if bb_ray_prescreen_results["surface_snap_point"] is None:
        # no support under this object, return failure
        return False

    # finish up
    if bb_ray_prescreen_results["surface_snap_point"] is not None:
        # accept the final location if a valid location exists
        obj.translation = bb_ray_prescreen_results["surface_snap_point"]
        if vdb is not None:
            vdb.get_observation(obj.translation)
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


def get_all_object_ids(sim):
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


def cull_string_list_by_substrings(
    full_list: List[str],
    included_substrings: List[str],
    excluded_substrings: List[str],
):
    """
    Cull a list of strings to the subset including any of the "included_substrings" and none of the excluded substrings.
    Returns the culled list, does not modify the input list.
    """
    culled_list: List[str] = []
    for string in full_list:
        excluded = False
        for excluded_substring in excluded_substrings:
            if excluded_substring in string:
                excluded = True
                break
        if not excluded:
            for included_substring in included_substrings:
                if included_substring in string:
                    culled_list.append(string)
                    break
    return culled_list
