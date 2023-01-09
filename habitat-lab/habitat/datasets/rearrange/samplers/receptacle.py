#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import magnum as mn
import numpy as np
import trimesh

import habitat_sim
from habitat.core.logging import logger
from habitat.sims.habitat_simulator.sim_utilities import add_wire_box


class Receptacle(ABC):
    """
    Defines a volume or surface for sampling object placements within a scene.
    Receptacles can be attached to rigid and articulated objects or defined in the global space of a stage or scene.
    Receptacle metadata should be defined in the SceneDataset in object_config.json, ao_config.json, and stage_config.json, and scene_config.json files or added programmatically to associated Attributes objects.
    To define a Receptacle within a JSON metadata file, add a new subgroup with a key beginning with "receptacle_" to the "user_defined" JSON subgroup. See ReplicaCAD v1.2+ for examples.
    """

    def __init__(
        self,
        name: str,
        parent_object_handle: str = None,
        parent_link: Optional[int] = None,
        up: Optional[mn.Vector3] = None,
    ):
        """
        :param name: The name of the Receptacle. Should be unique and descriptive for any one object.
        :param parent_object_handle: The rigid or articulated object instance handle for the parent object to which the Receptacle is attached. None for globally defined stage Receptacles.
        :param parent_link: Index of the link to which the Receptacle is attached if the parent is an ArticulatedObject. -1 denotes the base link. None for rigid objects and stage Receptables.
        :param up: The "up" direction of the receptacle in local AABB space. Used for optionally culling receptacles in un-supportive states such as inverted surfaces.
        """
        self.name = name
        self.up = (
            up if up is not None else mn.Vector3.y_axis(1.0)
        )  # default local Y up
        nonzero_indices = np.nonzero(self.up)
        assert (
            len(nonzero_indices) == 1
        ), "The 'up' vector must be aligned with a primary axis for an AABB."
        self.up_axis = nonzero_indices[0]
        self.parent_object_handle = parent_object_handle
        self.parent_link = parent_link

    @property
    def is_parent_object_articulated(self):
        """
        Convenience query for articulated vs. rigid object check.
        """
        return self.parent_link is not None

    @abstractmethod
    def sample_uniform_local(
        self, sample_region_scale: float = 1.0
    ) -> mn.Vector3:
        """
        Sample a uniform random point within Receptacle in local space.

        :param sample_region_scale: defines a XZ scaling of the sample region around its center. For example to constrain object spawning toward the center of a receptacle.
        """

    def get_global_transform(self, sim: habitat_sim.Simulator) -> mn.Matrix4:
        """
        Isolates boilerplate necessary to extract receptacle global transform of the Receptacle at the current state.
        """
        if self.parent_object_handle is None:
            # global identify by default
            return mn.Matrix4.identity_init()
        elif not self.is_parent_object_articulated:
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

    def get_local_transform(self, sim: habitat_sim.Simulator) -> mn.Matrix4:
        return self.get_global_transform(sim).inverted()

    def get_surface_center(self, sim: habitat_sim.Simulator) -> mn.Vector3:
        """
        Returns the center of receptacle surface in world space
        """
        local_center = self.get_local_surface_center(sim)
        return self.get_global_transform(sim).transform_point(local_center)

    @abstractmethod
    def get_local_surface_center(
        self, sim: habitat_sim.Simulator
    ) -> mn.Vector3:
        """
        Returns the center of receptacle surface in local space
        """

    def check_if_point_on_surface(
        self,
        sim: habitat_sim.Simulator,
        point: mn.Vector3,
        threshold: float = 0.05,
    ) -> bool:
        """
        Check if point lies within a `threshold` distance of the receptacle's surface
        """

    def sample_uniform_global(
        self, sim: habitat_sim.Simulator, sample_region_scale: float
    ) -> mn.Vector3:
        """
        Sample a uniform random point in the local Receptacle volume and then transform it into global space.

        :param sample_region_scale: defines a XZ scaling of the sample region around its center.
        """
        local_sample = self.sample_uniform_local(sample_region_scale)
        return self.get_global_transform(sim).transform_point(local_sample)

    def add_receptacle_visualization(
        self, sim: habitat_sim.Simulator
    ) -> List[habitat_sim.physics.ManagedRigidObject]:
        """
        Add one or more visualization objects to the simulation to represent the Receptacle. Return and forget the added objects for external management.
        """
        return []

    @abstractmethod
    def debug_draw(self, sim, color=None) -> None:
        """
        Render the Receptacle with DebugLineRender utility at the current frame.
        Simulator must be provided. If color is provided, the debug render will use it.
        Must be called after each frame is rendered, before querying the image data.
        """


class OnTopOfReceptacle(Receptacle):
    def __init__(self, name: str, places: List[str]):
        super().__init__(name)
        self._places = places

    def set_episode_data(self, episode_data):
        self.episode_data = episode_data

    def sample_uniform_local(
        self, sample_region_scale: float = 1.0
    ) -> mn.Vector3:
        return mn.Vector3(0.0, 0.1, 0.0)

    def get_global_transform(self, sim: habitat_sim.Simulator) -> mn.Matrix4:
        targ_T = list(self.episode_data["sampled_targets"].values())[0]
        # sampled_obj = self.episode_data["sampled_objects"][self._places[0]][0]
        # return sampled_obj.transformation

        return mn.Matrix4([[targ_T[j][i] for j in range(4)] for i in range(4)])

    def debug_draw(self, sim, color=None) -> None:
        """
        Render the Receptacle with DebugLineRender utility at the current frame.
        Simulator must be provided. If color is provided, the debug render will use it.
        Must be called after each frame is rendered, before querying the image data.
        """
        # TODO:


class AABBReceptacle(Receptacle):
    """
    Defines an AABB Receptacle volume above a surface for sampling object placements within a scene.
    """

    def __init__(
        self,
        name: str,
        bounds: mn.Range3D,
        parent_object_handle: str = None,
        parent_link: Optional[int] = None,
        up: Optional[mn.Vector3] = None,
        rotation: Optional[mn.Quaternion] = None,
    ) -> None:
        """
        :param name: The name of the Receptacle. Should be unique and descriptive for any one object.
        :param bounds: The AABB of the Receptacle.
        :param up: The "up" direction of the Receptacle in local AABB space. Used for optionally culling receptacles in un-supportive states such as inverted surfaces.
        :param parent_object_handle: The rigid or articulated object instance handle for the parent object to which the Receptacle is attached. None for globally defined stage Receptacles.
        :param parent_link: Index of the link to which the Receptacle is attached if the parent is an ArticulatedObject. -1 denotes the base link. None for rigid objects and stage Receptables.
        :param rotation: Optional rotation of the Receptacle AABB. Only used for globally defined stage Receptacles to provide flexability.
        """
        super().__init__(name, parent_object_handle, parent_link, up)
        self.bounds = bounds
        self.rotation = rotation if rotation is not None else mn.Quaternion()

    def sample_uniform_local(
        self, sample_region_scale: float = 1.0
    ) -> mn.Vector3:
        """
        Sample a uniform random point in the local AABB.

        :param sample_region_scale: defines a XZ scaling of the sample region around its center. For example to constrain object spawning toward the center of a receptacle.
        """
        scaled_region = mn.Range3D.from_center(
            self.bounds.center(), sample_region_scale * self.bounds.size() / 2
        )

        # NOTE: does not scale the "up" direction
        sample_range = [scaled_region.min, scaled_region.max]
        sample_range[0][self.up_axis] = self.bounds.min[self.up_axis]
        sample_range[1][self.up_axis] = self.bounds.max[self.up_axis]

        return np.random.uniform(sample_range[0], sample_range[1])

    def get_global_transform(self, sim: habitat_sim.Simulator) -> mn.Matrix4:
        """
        Isolates boilerplate necessary to extract receptacle global transform of the Receptacle at the current state.
        This specialization adds override rotation handling for global bounding box Receptacles.
        """
        if self.parent_object_handle is None:
            # this is a global stage receptacle
            from habitat_sim.utils.common import quat_from_two_vectors as qf2v
            from habitat_sim.utils.common import quat_to_magnum as qtm

            # TODO: add an API query or other method to avoid reconstructing the stage frame here
            stage_config = sim.get_stage_initialization_template()
            r_frameup_worldup = qf2v(
                habitat_sim.geo.UP, stage_config.orient_up
            )
            v_prime = qtm(r_frameup_worldup).transform_vector(
                mn.Vector3(habitat_sim.geo.FRONT)
            )
            world_to_local = (
                qf2v(np.array(v_prime), np.array(stage_config.orient_front))
                * r_frameup_worldup
            )
            world_to_local = habitat_sim.utils.common.quat_to_magnum(
                world_to_local
            )
            local_to_world = world_to_local.inverted()
            l2w4 = mn.Matrix4.from_(local_to_world.to_matrix(), mn.Vector3())

            # apply the receptacle rotation from the bb center
            T = mn.Matrix4.from_(mn.Matrix3(), self.bounds.center())
            R = mn.Matrix4.from_(self.rotation.to_matrix(), mn.Vector3())
            # translate frame to center, rotate, translate back
            l2w4 = l2w4.__matmul__(T.__matmul__(R).__matmul__(T.inverted()))
            return l2w4

        # base class implements getting transform from attached objects
        return super().get_global_transform

    def get_local_surface_center(
        self, sim: habitat_sim.Simulator
    ) -> mn.Vector3:
        local_center = self.bounds.center()
        local_center.y = self.bounds.y().min
        return local_center

    def check_if_point_on_surface(
        self,
        sim: habitat_sim.Simulator,
        point: mn.Vector3,
        threshold: float = 0.05,
    ) -> bool:
        local_point = self.get_local_transform(sim).transform_point(point)
        bounds = self.bounds
        return (
            bounds.x().contains(local_point.x)
            and bounds.z().contains(local_point.z)
            and np.abs(bounds.y().min - local_point.y) < threshold
        )

    def add_receptacle_visualization(
        self, sim: habitat_sim.Simulator
    ) -> List[habitat_sim.physics.ManagedRigidObject]:
        """
        Add a wireframe box object to the simulation to represent the AABBReceptacle and return it for external management.
        """
        attachment_scene_node = None
        if self.is_parent_object_articulated:
            attachment_scene_node = (
                sim.get_articulated_object_manager()
                .get_object_by_handle(self.parent_object_handle)
                .get_link_scene_node(self.parent_link)
                .create_child()
            )
        elif self.parent_object_handle is not None:
            # attach to the 1st visual scene node so any COM shift is automatically applied
            attachment_scene_node = (
                sim.get_rigid_object_manager()
                .get_object_by_handle(self.parent_object_handle)
                .visual_scene_nodes[1]
                .create_child()
            )
        box_obj = add_wire_box(
            sim,
            self.bounds.size() / 2.0,
            self.bounds.center(),
            attach_to=attachment_scene_node,
        )
        # TODO: enable rotation for object local receptacles

        # handle local frame and rotation for global receptacles
        if self.parent_object_handle is None:
            box_obj.transformation = self.get_global_transform(sim).__matmul__(
                box_obj.transformation
            )
        return [box_obj]

    def debug_draw(self, sim, color=None):
        """
        Render the AABBReceptacle with DebugLineRender utility at the current frame.
        Simulator must be provided. If color is provided, the debug render will use it.
        Must be called after each frame is rendered, before querying the image data.
        """
        # draw the box
        if color is None:
            color = mn.Color4.magenta()
        dblr = sim.get_debug_line_render()
        dblr.push_transform(self.get_global_transform(sim))
        dblr.draw_box(self.bounds.min, self.bounds.max, color)
        dblr.pop_transform()
        # TODO: test this


class TriangleMeshReceptacle(Receptacle):
    """
    Defines a Receptacle surface as a triangle mesh.
    TODO: configurable maximum height.
    """

    def __init__(
        self,
        name: str,
        mesh_data: Tuple[List[Any], List[Any]],  # vertices, indices
        parent_object_handle: str = None,
        parent_link: Optional[int] = None,
        up: Optional[mn.Vector3] = None,
    ) -> None:
        """
        :param name: The name of the Receptacle. Should be unique and descriptive for any one object.
        :param up: The "up" direction of the Receptacle in local AABB space. Used for optionally culling receptacles in un-supportive states such as inverted surfaces.
        :param parent_object_handle: The rigid or articulated object instance handle for the parent object to which the Receptacle is attached. None for globally defined stage Receptacles.
        :param parent_link: Index of the link to which the Receptacle is attached if the parent is an ArticulatedObject. -1 denotes the base link. None for rigid objects and stage Receptables.
        """
        super().__init__(name, parent_object_handle, parent_link, up)
        self.mesh_data = mesh_data
        self.area_weighted_accumulator = (
            []
        )  # normalized float weights for each triangle for sampling
        assert len(mesh_data[1]) % 3 == 0, "must be triangles"
        self.total_area = 0
        triangles = []
        for f_ix in range(int(len(mesh_data[1]) / 3)):
            v = self.get_face_verts(f_ix)
            w1 = v[1] - v[0]
            w2 = v[2] - v[1]
            triangles.append(v)
            self.area_weighted_accumulator.append(
                0.5 * np.linalg.norm(np.cross(w1, w2))
            )
            self.total_area += self.area_weighted_accumulator[-1]
        for f_ix in range(len(self.area_weighted_accumulator)):
            self.area_weighted_accumulator[f_ix] = (
                self.area_weighted_accumulator[f_ix] / self.total_area
            )
            if f_ix > 0:
                self.area_weighted_accumulator[
                    f_ix
                ] += self.area_weighted_accumulator[f_ix - 1]
        self.trimesh = trimesh.Trimesh(
            **trimesh.triangles.to_kwargs(triangles)
        )

    def get_face_verts(self, f_ix):
        verts = []
        for ix in range(3):
            verts.append(
                np.array(
                    self.mesh_data[0][self.mesh_data[1][int(f_ix * 3 + ix)]]
                )
            )
        return verts

    def sample_area_weighted_triangle(self):
        """
        Isolates the area weighted triangle sampling code.
        """

        def find_ge(a, x):
            "Find leftmost item greater than or equal to x"
            from bisect import bisect_left

            i = bisect_left(a, x)
            if i != len(a):
                return i
            raise ValueError

        # first area weighted sampling of a triangle
        sample_val = random.random()
        tri_index = find_ge(self.area_weighted_accumulator, sample_val)
        return tri_index

    def get_local_surface_center(
        self, sim: habitat_sim.Simulator
    ) -> mn.Vector3:
        return self.trimesh.centroid

    def check_if_point_on_surface(
        self,
        sim: habitat_sim.Simulator,
        point: mn.Vector3,
        threshold: float = 0.05,
    ) -> bool:
        return (
            np.abs(trimesh.proximity.signed_distance(self.trimesh, [point]))
            < threshold
        )

    def sample_uniform_local(
        self, sample_region_scale: float = 1.0
    ) -> mn.Vector3:
        """
        Sample a uniform random point from the mesh.

        :param sample_region_scale: defines a XZ scaling of the sample region around its center. For example to constrain object spawning toward the center of a receptacle.
        """

        if sample_region_scale != 1.0:
            logger.warning(
                "TriangleMeshReceptacle does not support 'sample_region_scale' != 1.0."
            )

        tri_index = self.sample_area_weighted_triangle()

        # then sample a random point in the triangle
        # https://math.stackexchange.com/questions/538458/how-to-sample-points-on-a-triangle-surface-in-3d
        coef1 = random.random()
        coef2 = random.random()
        if coef1 + coef2 >= 1:
            coef1 = 1 - coef1
            coef2 = 1 - coef2
        v = self.get_face_verts(f_ix=tri_index)
        rand_point = v[0] + coef1 * (v[1] - v[0]) + coef2 * (v[2] - v[0])

        return rand_point

    def debug_draw(self, sim, color=None):
        """
        Render the Receptacle with DebugLineRender utility at the current frame.
        Draws the Receptacle mesh.
        Simulator must be provided. If color is provided, the debug render will use it.
        Must be called after each frame is rendered, before querying the image data.
        """
        # draw all mesh triangles
        if color is None:
            color = mn.Color4.magenta()
        dblr = sim.get_debug_line_render()
        assert len(self.mesh_data[1]) % 3 == 0, "must be triangles"
        for face in range(int(len(self.mesh_data[1]) / 3)):
            verts = self.get_face_verts(f_ix=face)
            for edge in range(3):
                dblr.draw_transformed_line(
                    verts[edge], verts[(edge + 1) % 3], color
                )


def get_all_scenedataset_receptacles(sim) -> Dict[str, Dict[str, List[str]]]:
    """
    Scrapes the active SceneDataset from a Simulator for all receptacle names defined in rigid/articulated object and stage templates for investigation and preview purposes.
    Note this will not include scene-specific overrides defined in scene_config.json files. Only receptacles defined in object_config.json, ao_config.json, and stage_config.json files or added programmatically to associated Attributes objects will be found.

    Returns a dict with keys {"stage", "rigid", "articulated"} mapping object template handles to lists of receptacle names.
    """
    # cache the rigid and articulated receptacles seperately
    receptacles: Dict[str, Dict[str, List[str]]] = {
        "stage": {},
        "rigid": {},
        "articulated": {},
    }

    # scrape stage configs:
    stm = sim.get_stage_template_manager()
    for template_handle in stm.get_template_handles(""):
        stage_template = stm.get_template_by_handle(template_handle)
        for item in stage_template.get_user_config().get_subconfig_keys():
            if item.startswith("receptacle_"):
                print(
                    f"template file_directory = {stage_template.file_directory}"
                )
                if template_handle not in receptacles["stage"]:
                    receptacles["stage"][template_handle] = []
                receptacles["stage"][template_handle].append(item)

    # scrape the rigid object configs:
    rotm = sim.get_object_template_manager()
    for template_handle in rotm.get_template_handles(""):
        obj_template = rotm.get_template_by_handle(template_handle)
        for item in obj_template.get_user_config().get_subconfig_keys():
            if item.startswith("receptacle_"):
                print(
                    f"template file_directory = {obj_template.file_directory}"
                )
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


def import_tri_mesh_ply(ply_file: str) -> Tuple[List[mn.Vector3], List[int]]:
    """
    Returns a Tuple of (verts,indices) from a ply mesh.
    NOTE: the input PLY must contain only triangles.
    TODO: This could be replaced by a standard importer, but I didn't want to add additional dependencies for such as small feature.
    """
    mesh_data: Tuple[List[mn.Vector3], List[int]] = ([], [])
    with open(ply_file) as f:
        lines = [line.rstrip() for line in f]
        assert lines[0] == "ply", f"Must be PLY format. '{ply_file}'"
        assert "format ascii" in lines[1], f"Must be ascii PLY. '{ply_file}'"
        # parse the header
        line_index = 2
        num_verts = 0
        num_faces = 0
        while line_index < len(lines):
            if lines[line_index].startswith("element vertex"):
                num_verts = int(lines[line_index][14:])
                print(f"num_verts = {num_verts}")
            elif lines[line_index].startswith("element face"):
                num_faces = int(lines[line_index][12:])
                print(f"num_faces = {num_faces}")
            elif lines[line_index] == "end_header":
                # done parsing header
                line_index += 1
                break
            line_index += 1
        assert (
            len(lines) - line_index == num_verts + num_faces
        ), f"Lines after header ({len(lines) - line_index}) should agree with forward declared content. {num_verts} verts and {num_faces} faces expected. '{ply_file}'"
        # parse the verts
        for vert_line in range(line_index, num_verts + line_index):
            coords = [float(x) for x in lines[vert_line].split(" ")]
            mesh_data[0].append(mn.Vector3(coords))
        line_index += num_verts
        for face_line in range(line_index, num_faces + line_index):
            assert (
                int(lines[face_line][0]) <= 4
            ), f"Faces must be triangles. '{ply_file}' {lines[face_line][0]}"
            if int(lines[face_line][0]) == 4:
                indices = [int(x) for x in lines[face_line].split(" ")[1:]]
                mesh_data[1].extend(indices[:-1])
                mesh_data[1].extend(indices[1:])
            else:
                indices = [int(x) for x in lines[face_line].split(" ")[1:]]
                mesh_data[1].extend(indices)

    return mesh_data


def parse_receptacles_from_user_config(
    user_subconfig: habitat_sim._ext.habitat_sim_bindings.Configuration,
    parent_object_handle: Optional[str] = None,
    parent_template_directory: str = "",
    valid_link_names: Optional[List[str]] = None,
    ao_uniform_scaling: float = 1.0,
) -> List[Union[Receptacle, AABBReceptacle]]:
    """
    Parse receptacle metadata from the provided user subconfig object.

    :param user_subconfig: The Configuration object containing metadata parsed from the "user_defined" JSON field for rigid/articulated object and stage configs.
    :param parent_object_handle: The instance handle of the rigid or articulated object to which constructed Receptacles are attached. None or globally defined stage Receptacles.
    :param valid_link_names: An indexed list of link names for validating configured Receptacle attachments. Provided only for ArticulatedObjects.
    :param valid_link_names: An indexed list of link names for validating configured Receptacle attachments. Provided only for ArticulatedObjects.
    :param ao_uniform_scaling: Uniform scaling applied to the parent AO is applied directly to the Receptacle.

    Construct and return a list of Receptacle objects. Multiple Receptacles can be defined in a single user subconfig.
    """
    receptacles: List[
        Union[Receptacle, AABBReceptacle, TriangleMeshReceptacle]
    ] = []

    # pre-define unique specifier strings for parsing receptacle types
    receptacle_prefix_string = "receptacle_"
    mesh_receptacle_id_string = "receptacle_mesh_"
    aabb_receptacle_id_string = "receptacle_aabb_"
    # search the generic user subconfig metadata looking for receptacles
    for sub_config_key in user_subconfig.get_subconfig_keys():
        if sub_config_key.startswith(receptacle_prefix_string):
            sub_config = user_subconfig.get_subconfig(sub_config_key)
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

            # optional rotation for global receptacles, defaults to identity
            rotation = (
                mn.Quaternion()
                if not sub_config.has_value("rotation")
                else sub_config.get("rotation")
            )

            # setup parent specific metadata for ArticulatedObjects
            parent_link_ix = None
            if valid_link_names is not None:
                assert sub_config.has_value(
                    "parent_link"
                ), "ArticulatedObject Receptacles must define a parent link name."
                parent_link_name = sub_config.get("parent_link")
                # search for a matching link
                for link_ix, link_name in enumerate(valid_link_names):
                    if link_name == parent_link_name:
                        parent_link_ix = (
                            link_ix - 1
                        )  # starting from -1 (base link)
                        break
                assert (
                    parent_link_ix is not None
                ), f"('parent_link' = '{parent_link_name}') in Receptacle configuration does not match any provided link names: {valid_link_names}."
            else:
                assert not sub_config.has_value(
                    "parent_link"
                ), "ArticulatedObject parent link name defined in config, but no valid_link_names provided. Mistake?"

            # apply AO uniform instance scaling
            receptacle_position = ao_uniform_scaling * sub_config.get(
                "position"
            )
            receptacle_scale = ao_uniform_scaling * sub_config.get("scale")

            if aabb_receptacle_id_string in sub_config_key:
                receptacles.append(
                    AABBReceptacle(
                        name=receptacle_name,
                        bounds=mn.Range3D.from_center(
                            receptacle_position,
                            receptacle_scale,
                        ),
                        rotation=rotation,
                        up=up,
                        parent_object_handle=parent_object_handle,
                        parent_link=parent_link_ix,
                    )
                )
            elif mesh_receptacle_id_string in sub_config_key:
                mesh_file = os.path.join(
                    parent_template_directory, sub_config.get("mesh_filepath")
                )
                assert os.path.exists(
                    mesh_file
                ), f"Configured receptacle mesh asset '{mesh_file}' not found."
                # TODO: build the mesh_data entry from scale and mesh
                mesh_data = import_tri_mesh_ply(mesh_file)

                receptacles.append(
                    TriangleMeshReceptacle(
                        name=receptacle_name,
                        mesh_data=mesh_data,
                        up=up,
                        parent_object_handle=parent_object_handle,
                        parent_link=parent_link_ix,
                    )
                )
            else:
                raise AssertionError(
                    f"Receptacle detected without a subtype specifier: '{mesh_receptacle_id_string}'"
                )
    return receptacles


def find_receptacles(
    sim: habitat_sim.Simulator,
) -> List[Union[Receptacle, AABBReceptacle]]:
    """
    Scrape and return a list of all Receptacles defined in the metadata belonging to the scene's currently instanced objects.
    """

    obj_mgr = sim.get_rigid_object_manager()
    ao_mgr = sim.get_articulated_object_manager()

    receptacles: List[
        Union[Receptacle, AABBReceptacle, TriangleMeshReceptacle]
    ] = []

    # search for global receptacles included with the stage
    stage_config = sim.get_stage_initialization_template()
    if stage_config is not None:
        stage_user_attr = stage_config.get_user_config()
        receptacles.extend(
            parse_receptacles_from_user_config(
                stage_user_attr,
                parent_template_directory=stage_config.file_directory,
            )
        )

    # rigid object receptacles
    for obj_handle in obj_mgr.get_object_handles():
        obj = obj_mgr.get_object_by_handle(obj_handle)
        source_template_file = obj.creation_attributes.file_directory
        user_attr = obj.user_attributes
        receptacles.extend(
            parse_receptacles_from_user_config(
                user_attr,
                parent_object_handle=obj_handle,
                parent_template_directory=source_template_file,
            )
        )

    # articulated object receptacles
    for obj_handle in ao_mgr.get_object_handles():
        obj = ao_mgr.get_object_by_handle(obj_handle)
        # TODO: no way to get filepath from AO currently. Add this API.
        source_template_file = ""
        user_attr = obj.user_attributes
        receptacles.extend(
            parse_receptacles_from_user_config(
                user_attr,
                parent_object_handle=obj_handle,
                parent_template_directory=source_template_file,
                valid_link_names=[
                    obj.get_link_name(link)
                    for link in range(-1, obj.num_links)
                ],
                ao_uniform_scaling=obj.global_scale,
            )
        )

    return receptacles


@dataclass
class ReceptacleSet:
    name: str
    included_object_substrings: List[str]
    excluded_object_substrings: List[str]
    included_receptacle_substrings: List[str]
    excluded_receptacle_substrings: List[str]
    is_on_top_of_sampler: bool = False
    comment: str = ""


class ReceptacleTracker:
    def __init__(
        self,
        max_objects_per_receptacle,
        receptacle_sets: Dict[str, ReceptacleSet],
    ):
        self._receptacle_counts = dict(max_objects_per_receptacle)
        self._receptacle_sets = {
            k: deepcopy(v) for k, v in receptacle_sets.items()
        }

    @property
    def recep_sets(self) -> Dict[str, ReceptacleSet]:
        return self._receptacle_sets

    def inc_count(self, recep_name):
        if recep_name in self._receptacle_counts:
            self._receptacle_counts[recep_name] += 1

    def update_receptacle_tracking(self, new_receptacle: Receptacle):
        recep_name = new_receptacle.name
        if recep_name not in self._receptacle_counts:
            return False
        self._receptacle_counts[recep_name] -= 1
        if self._receptacle_counts[recep_name] < 0:
            raise ValueError(f"Receptacle count for {recep_name} is invalid")
        if self._receptacle_counts[recep_name] == 0:
            for receptacle_set in self._receptacle_sets.values():
                # Exclude this receptacle from appearing in the future.
                if (
                    recep_name
                    not in receptacle_set.excluded_receptacle_substrings
                ):
                    receptacle_set.excluded_receptacle_substrings.append(
                        recep_name
                    )
                if recep_name in receptacle_set.included_receptacle_substrings:
                    recep_idx = (
                        receptacle_set.included_receptacle_substrings.index(
                            recep_name
                        )
                    )
                    del receptacle_set.included_receptacle_substrings[
                        recep_idx
                    ]
            return True
        return False
