#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import corrade as cr
import magnum as mn
import numpy as np
import trimesh
from tqdm import tqdm

import habitat_sim
from habitat.core.logging import logger
from habitat.datasets.rearrange.navmesh_utils import is_accessible
from habitat.datasets.rearrange.viewpoints import generate_viewpoints
from habitat.sims.habitat_simulator.sim_utilities import add_wire_box
from habitat.tasks.nav.object_nav_task import ObjectViewLocation
from habitat.tasks.rearrange.utils import get_aabb
from habitat.utils.geometry_utils import random_triangle_point

# global module singleton for mesh importing instantiated upon first import
_manager = mn.trade.ImporterManager()


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

        # The unique name of this Receptacle instance in the current scene.
        # This name is a combination of the object instance name and Receptacle name.
        self.unique_name = ""
        if self.parent_object_handle is None:
            # this is a stage receptacle
            self.unique_name = "stage|" + self.name
        else:
            self.unique_name = self.parent_object_handle + "|" + self.name

    @property
    def is_parent_object_articulated(self):
        """
        Convenience query for articulated vs. rigid object check.
        """
        return self.parent_link is not None

    @property
    @abstractmethod
    def total_area(self) -> float:
        """
        Get total area of receptacle surface
        """

    @property
    @abstractmethod
    def bounds(self) -> mn.Range3D:
        """
        Get the bounds of the AABB of the receptacle
        """

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
        # handle global parent
        if self.parent_object_handle is None:
            # global identify by default
            return mn.Matrix4.identity_init()

        # handle RigidObject parent
        if not self.is_parent_object_articulated:
            obj_mgr = sim.get_rigid_object_manager()
            obj = obj_mgr.get_object_by_handle(self.parent_object_handle)
            # NOTE: we use absolute transformation from the 2nd visual node (scaling node) and root of all render assets to correctly account for any COM shifting, re-orienting, or scaling which has been applied.
            return obj.visual_scene_nodes[1].absolute_transformation()

        # handle ArticulatedObject parent
        ao_mgr = sim.get_articulated_object_manager()
        obj = ao_mgr.get_object_by_handle(self.parent_object_handle)
        return obj.get_link_scene_node(
            self.parent_link
        ).absolute_transformation()

    def get_local_transform(self, sim: habitat_sim.Simulator) -> mn.Matrix4:
        """
        Returns transformation that can be used for transforming from world space to receptacle's local space
        """
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

    @abstractmethod
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
    def debug_draw(
        self, sim: habitat_sim.Simulator, color: Optional[mn.Color4] = None
    ) -> None:
        """
        Render the Receptacle with DebugLineRender utility at the current frame.
        Must be called after each frame is rendered, before querying the image data.

        :param sim: Simulator must be provided.
        :param color: Optionally provide wireframe color, otherwise magenta.
        """
        raise NotImplementedError


class OnTopOfReceptacle(Receptacle):
    def __init__(self, name: str, places: List[str]):
        super().__init__(name)
        self._places = places

    @property
    def total_area(self) -> float:
        raise NotImplementedError

    @property
    def bounds(self) -> mn.Range3D:
        raise NotImplementedError

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

    def debug_draw(
        self, sim: habitat_sim.Simulator, color: Optional[mn.Color4] = None
    ) -> None:
        """
        Render the Receptacle with DebugLineRender utility at the current frame.
        Must be called after each frame is rendered, before querying the image data.

        :param sim: Simulator must be provided.
        :param color: Optionally provide wireframe color, otherwise magenta.
        """
        # TODO:

    def check_if_point_on_surface(
        self,
        sim: habitat_sim.Simulator,
        point: mn.Vector3,
        threshold: float = 0.05,
    ) -> bool:
        """
        Returns True if the point lies within the `threshold` distance of the lower bound along the "up" axis and within the bounds along other axes
        """
        # TODO:
        raise NotImplementedError

    def get_local_surface_center(
        self, sim: habitat_sim.Simulator
    ) -> mn.Vector3:
        """
        Returns the center of receptacle surface in local space
        """
        # TODO:
        raise NotImplementedError


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
        self._bounds = bounds
        self.rotation = rotation if rotation is not None else mn.Quaternion()

    @property
    def total_area(self) -> float:
        return self._bounds.size_x() * self._bounds.size_z()

    @property
    def bounds(self) -> mn.Range3D:
        return self._bounds

    def sample_uniform_local(
        self, sample_region_scale: float = 1.0
    ) -> mn.Vector3:
        """
        Sample a uniform random point in the local AABB.
        :param sample_region_scale: defines a XZ scaling of the sample region around its center. For example to constrain object spawning toward the center of a receptacle.
        """
        scaled_region = mn.Range3D.from_center(
            self._bounds.center(),
            sample_region_scale * self._bounds.size() / 2,
        )

        # NOTE: does not scale the "up" direction
        sample_range = [scaled_region.min, scaled_region.max]
        sample_range[0][self.up_axis] = self._bounds.min[self.up_axis]
        sample_range[1][self.up_axis] = self._bounds.max[self.up_axis]

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
            T = mn.Matrix4.from_(mn.Matrix3(), self._bounds.center())
            R = mn.Matrix4.from_(self.rotation.to_matrix(), mn.Vector3())
            # translate frame to center, rotate, translate back
            l2w4 = l2w4.__matmul__(T.__matmul__(R).__matmul__(T.inverted()))
            return l2w4

        # base class implements getting transform from attached objects
        return super().get_global_transform(sim)

    def get_local_surface_center(
        self, sim: habitat_sim.Simulator
    ) -> mn.Vector3:
        local_center = self._bounds.center()
        local_center[self.up_axis] = self._bounds.min[self.up_axis]
        return local_center

    def check_if_point_on_surface(
        self,
        sim: habitat_sim.Simulator,
        point: mn.Vector3,
        threshold: float = 0.05,
    ) -> bool:
        """
        Returns True if the point lies within the `threshold` distance of the lower bound along the "up" axis and within the bounds along other axes
        """
        local_point = self.get_local_transform(sim).transform_point(point)
        bounds = self._bounds
        on_surface = True
        bounds_min = bounds.min
        bounds_max = bounds.max
        for i in range(3):
            if i == self.up_axis:
                on_surface = (
                    on_surface
                    and np.abs(bounds_min[i] - local_point[i]) < threshold
                )
            else:
                on_surface = (
                    on_surface
                    and bounds_min[i] <= local_point[i]
                    and local_point[i] <= bounds_max[i]
                )
        return on_surface

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

    def debug_draw(
        self, sim: habitat_sim.Simulator, color: Optional[mn.Color4] = None
    ) -> None:
        """
        Render the AABBReceptacle with DebugLineRender utility at the current frame.
        Must be called after each frame is rendered, before querying the image data.

        :param sim: Simulator must be provided.
        :param color: Optionally provide wireframe color, otherwise magenta.
        """
        # draw the box
        if color is None:
            color = mn.Color4.magenta()
        dblr = sim.get_debug_line_render()
        dblr.push_transform(self.get_global_transform(sim))
        dblr.draw_box(self.bounds.min, self.bounds.max, color)
        dblr.pop_transform()
        # TODO: test this


def assert_triangles(indices: List[int]) -> None:
    """
    Assert that an index array is divisible by 3 as a heuristic for triangle-only faces.
    """
    assert (
        len(indices) % 3 == 0
    ), "TriangleMeshReceptacles must be exclusively composed of triangles. The provided mesh_data is not."


class TriangleMeshReceptacle(Receptacle):
    """
    Defines a Receptacle surface as a triangle mesh.
    TODO: configurable maximum height.
    """

    def __init__(
        self,
        name: str,
        mesh_data: mn.trade.MeshData,
        parent_object_handle: str = None,
        parent_link: Optional[int] = None,
        up: Optional[mn.Vector3] = None,
    ) -> None:
        """
        Initialize the TriangleMeshReceptacle from mesh data and pre-compute the area weighted accumulator.

        :param name: The name of the Receptacle. Should be unique and descriptive for any one object.
        :param mesh_data: The Receptacle's mesh data. A magnum.trade.MeshData object (indices len divisible by 3).
        :param parent_object_handle: The rigid or articulated object instance handle for the parent object to which the Receptacle is attached. None for globally defined stage Receptacles.
        :param parent_link: Index of the link to which the Receptacle is attached if the parent is an ArticulatedObject. -1 denotes the base link. None for rigid objects and stage Receptables.
        :param up: The "up" direction of the Receptacle in local AABB space. Used for optionally culling receptacles in un-supportive states such as inverted surfaces.
        """
        super().__init__(name, parent_object_handle, parent_link, up)
        self.mesh_data = mesh_data
        self.area_weighted_accumulator = (
            []
        )  # normalized float weights for each triangle for sampling
        assert_triangles(mesh_data.indices)

        # pre-compute the normalized cumulative area of all triangle faces for later sampling
        self._total_area = 0
        triangles = []
        for f_ix in range(int(len(mesh_data.indices) / 3)):
            v = self.get_face_verts(f_ix)
            w1 = v[1] - v[0]
            w2 = v[2] - v[1]
            triangles.append(v)
            self.area_weighted_accumulator.append(
                0.5 * mn.math.cross(w1, w2).length()
            )
            self._total_area += self.area_weighted_accumulator[-1]
        for f_ix in range(len(self.area_weighted_accumulator)):
            self.area_weighted_accumulator[f_ix] = (
                self.area_weighted_accumulator[f_ix] / self._total_area
            )
            if f_ix > 0:
                self.area_weighted_accumulator[
                    f_ix
                ] += self.area_weighted_accumulator[f_ix - 1]

        # TODO: Remove dependency on trimesh
        self.trimesh = trimesh.Trimesh(
            **trimesh.triangles.to_kwargs(triangles)
        )

    @property
    def total_area(self) -> float:
        return self._total_area

    @property
    def bounds(self) -> mn.Range3D:
        return mn.Range3D(self.trimesh.bounds)

    def get_face_verts(self, f_ix: int) -> List[mn.Vector3]:
        """
        Get all three vertices of a mesh triangle given it's face index as a list of numpy arrays.

        :param f_ix: The index of the mesh triangle.
        """
        verts: List[mn.Vector3] = []
        for ix in range(3):
            index = int(f_ix * 3 + ix)
            v_ix = self.mesh_data.indices[index]
            verts.append(
                self.mesh_data.attribute(mn.trade.MeshAttribute.POSITION)[v_ix]
            )
        return verts

    def sample_area_weighted_triangle(self) -> int:
        """
        Isolates the area weighted triangle sampling code.

        Returns a random triangle index sampled with area weighting.
        """

        def find_ge(a: List[Any], x) -> Any:
            "Find leftmost item greater than or equal to x"
            from bisect import bisect_left

            i = bisect_left(a, x)
            if i != len(a):
                return i
            raise ValueError(
                f"Value '{x}' is greater than all items in the list. Maximum value should be <1."
            )

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
        local_point = self.get_local_transform(sim).transform_point(point)
        return (
            np.abs(
                trimesh.proximity.signed_distance(self.trimesh, [local_point])
            )
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
        v = self.get_face_verts(f_ix=tri_index)
        rand_point = random_triangle_point(v[0], v[1], v[2])

        return rand_point

    def debug_draw(
        self, sim: habitat_sim.Simulator, color: Optional[mn.Color4] = None
    ) -> None:
        """
        Render the Receptacle with DebugLineRender utility at the current frame.
        Draws the Receptacle mesh.
        Must be called after each frame is rendered, before querying the image data.

        :param sim: Simulator must be provided.
        :param color: Optionally provide wireframe color, otherwise magenta.
        """
        # draw all mesh triangles
        if color is None:
            color = mn.Color4.magenta()
        dblr = sim.get_debug_line_render()
        dblr.push_transform(self.get_global_transform(sim))
        assert_triangles(self.mesh_data.indices)
        for face in range(int(len(self.mesh_data.indices) / 3)):
            verts = self.get_face_verts(f_ix=face)
            for edge in range(3):
                dblr.draw_transformed_line(
                    verts[edge], verts[(edge + 1) % 3], color
                )
        dblr.pop_transform()


def get_all_scenedataset_receptacles(
    sim: habitat_sim.Simulator,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Scrapes the active SceneDataset from a Simulator for all receptacle names defined in rigid/articulated object and stage templates for investigation and preview purposes.
    Note this will not include scene-specific overrides defined in scene_config.json files. Only receptacles defined in object_config.json, ao_config.json, and stage_config.json files or added programmatically to associated Attributes objects will be found.
    Returns a dict with keys {"stage", "rigid", "articulated"} mapping object template handles to lists of receptacle names.

    :param sim: Simulator must be provided.
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


def filter_interleave_mesh(mesh: mn.trade.MeshData) -> mn.trade.MeshData:
    """
    Filter all but position data and interleave a mesh to reduce overall memory footprint.
    Convert triangle like primitives into triangles and assert only triangles remain.

    NOTE: Modifies the mesh data in-place
    :return: The modified mesh for easy of use.
    """

    # convert to triangles and validate the result
    if mesh.primitive in [
        mn.MeshPrimitive.TRIANGLE_STRIP,
        mn.MeshPrimitive.TRIANGLE_FAN,
    ]:
        mesh = mn.meshtools.generate_indices(mesh)
    assert (
        mesh.primitive == mn.MeshPrimitive.TRIANGLES
    ), "Must be a triangle mesh."

    # filter out all but positions (and indices) from the mesh
    mesh = mn.meshtools.filter_only_attributes(
        mesh, [mn.trade.MeshAttribute.POSITION]
    )

    # reformat the mesh data after filtering
    mesh = mn.meshtools.interleave(mesh, mn.meshtools.InterleaveFlags.NONE)

    return mesh


def import_tri_mesh(mesh_file: str) -> List[mn.trade.MeshData]:
    """
    Returns a list of MeshData objects from a mesh asset using magnum trade importer.

    :param mesh_file: The input meshes file. NOTE: must contain only triangles.
    """
    importer = _manager.load_and_instantiate("AnySceneImporter")
    importer.open_file(mesh_file)

    mesh_data: List[mn.trade.MeshData] = []

    # import mesh data and pre-process
    mesh_data = [
        filter_interleave_mesh(importer.mesh(mesh_ix))
        for mesh_ix in range(importer.mesh_count)
    ]

    # if there is a scene defined, apply any transformations
    if importer.scene_count > 0:
        scene_id = importer.default_scene
        # If there's no default scene, load the first one
        if scene_id == -1:
            scene_id = 0

        scene = importer.scene(scene_id)

        # Mesh referenced by mesh_assignments[i] has a corresponding transform in
        # mesh_transformations[i]. Association to a particular node ID is stored in
        # scene.mapping(mn.trade.SceneField.MESH)[i], but it's not needed for anything
        # here.
        mesh_assignments: cr.containers.StridedArrayView1D = scene.field(
            mn.trade.SceneField.MESH
        )

        mesh_transformations: List[
            mn.Matrix4
        ] = mn.scenetools.absolute_field_transformations3d(
            scene, mn.trade.SceneField.MESH
        )
        assert len(mesh_assignments) == len(mesh_transformations)

        # A mesh can be referenced by multiple nodes, so this can't operate in-place.
        # i.e., len(mesh_data) likely changes after this step
        mesh_data = [
            mn.meshtools.transform3d(mesh_data[mesh_id], transformation)
            for mesh_id, transformation in zip(
                mesh_assignments, mesh_transformations
            )
        ]

    return mesh_data


def parse_receptacles_from_user_config(
    user_subconfig: habitat_sim._ext.habitat_sim_bindings.Configuration,
    parent_object_handle: Optional[str] = None,
    parent_template_directory: str = "",
    valid_link_names: Optional[List[str]] = None,
    ao_uniform_scaling: float = 1.0,
) -> List[Union[Receptacle, AABBReceptacle, TriangleMeshReceptacle]]:
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
                mesh_data: List[mn.trade.MeshData] = import_tri_mesh(mesh_file)

                for mix, single_mesh_data in enumerate(mesh_data):
                    single_receptacle_name = (
                        receptacle_name + "." + str(mix).rjust(4, "0")
                    )
                    receptacles.append(
                        TriangleMeshReceptacle(
                            name=single_receptacle_name,
                            mesh_data=single_mesh_data,
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


def get_obj_manager_for_receptacle(
    sim: habitat_sim.Simulator, receptacle: Receptacle
):
    if receptacle.is_parent_object_articulated:
        obj_mgr = sim.get_articulated_object_manager()
    else:
        obj_mgr = sim.get_rigid_object_manager()
    return obj_mgr


def get_navigable_receptacles(
    sim: habitat_sim.Simulator,
    receptacles: List[Receptacle],
) -> List[Receptacle]:
    """Given a list of receptacles, return the ones that are navigable from the given navmesh island"""
    navigable_receptacles: List[Receptacle] = []
    for receptacle in receptacles:
        obj_mgr = get_obj_manager_for_receptacle(sim, receptacle)
        receptacle_obj = obj_mgr.get_object_by_handle(
            receptacle.parent_object_handle
        )
        receptacle_bb = get_aabb(
            receptacle_obj.object_id, sim, transformed=True
        )

        if (
            receptacle_bb.size_y()
            > sim.pathfinder.nav_mesh_settings.agent_height - 0.2
        ):
            logger.info(
                f"Receptacle {receptacle.parent_object_handle}, {receptacle_obj.translation} is too tall. Skipping."
            )
            continue

        bounds = receptacle.bounds
        if bounds.size_x() < 0.3 or bounds.size_z() < 0.3:
            logger.info(
                f"Receptacle {receptacle.parent_object_handle}, {receptacle_obj.translation} is too small. Skipping."
            )
            continue

        recep_points = [
            receptacle_bb.back_bottom_left,
            receptacle_bb.back_bottom_right,
            receptacle_bb.front_bottom_left,
            receptacle_bb.front_bottom_right,
        ]
        # At least 2 corners should be accessible
        num_corners_accessible = sum(
            is_accessible(sim, point, nav_to_min_distance=1.5)
            for point in recep_points
        )

        if not num_corners_accessible >= 2:
            logger.info(
                f"Receptacle {receptacle.parent_object_handle}, {receptacle_obj.translation} is not accessible. "
                f"Number of corners accessible: {num_corners_accessible}"
            )
            continue
        else:
            logger.info(
                f"Receptacle {receptacle.parent_object_handle}, {receptacle_obj.translation} is accessible."
            )
            navigable_receptacles.append(receptacle)

    logger.info(
        f"Found {len(navigable_receptacles)}/{len(receptacles)} accessible receptacles."
    )
    return navigable_receptacles


def get_receptacle_viewpoints(
    sim: habitat_sim.Simulator,
    receptacles: List[Receptacle],
    debug_viz: bool = False,
) -> Tuple[Dict[str, List[ObjectViewLocation]], List[Receptacle]]:
    viewpoints = {}
    viewable_receptacles = []
    logger.info("Getting receptacle viewpoints...")
    for receptacle in tqdm(receptacles):
        handle = receptacle.parent_object_handle
        if handle in viewpoints:
            continue
        obj_mgr = get_obj_manager_for_receptacle(sim, receptacle)
        receptacle_obj = obj_mgr.get_object_by_handle(handle)
        receptacle_viewpoints = generate_viewpoints(
            sim, receptacle_obj, debug_viz=debug_viz
        )
        if len(receptacle_viewpoints) > 0:
            viewpoints[handle] = receptacle_viewpoints
            viewable_receptacles.append(receptacle)
    return viewpoints, viewable_receptacles


def find_receptacles(
    sim: habitat_sim.Simulator,
) -> List[Union[Receptacle, AABBReceptacle, TriangleMeshReceptacle]]:
    """
    Scrape and return a list of all Receptacles defined in the metadata belonging to the scene's currently instanced objects.

    :param sim: Simulator must be provided.
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
        max_objects_per_receptacle: Dict[str, int],
        receptacle_sets: Dict[str, ReceptacleSet],
    ):
        """
        :param max_objects_per_receptacle: A Dict mapping receptacle unique names to the remaining number of objects allowed in the receptacle.
        :param receptacle_sets: Dict mapping ReceptacleSet name to its dataclass.
        """
        self._receptacle_counts: Dict[str, int] = max_objects_per_receptacle
        # deep copy ReceptacleSets because they may be modified by allocations
        self._receptacle_sets: Dict[str, ReceptacleSet] = {
            k: deepcopy(v) for k, v in receptacle_sets.items()
        }

    @property
    def recep_sets(self) -> Dict[str, ReceptacleSet]:
        return self._receptacle_sets

    def init_scene_filters(
        self, mm: habitat_sim.metadata.MetadataMediator, scene_handle: str
    ) -> None:
        """
        Initialize the scene specific filter strings from metadata.
        Looks for a filter file defined for the scene, loads filtered strings and adds them to the exclude list of all ReceptacleSets.

        :param mm: The active MetadataMediator instance from which to load the filter data.
        :param scene_handle: The handle of the currently instantiated scene.
        """
        scene_user_defined = mm.get_scene_user_defined(scene_handle)
        filtered_unique_names = []
        if scene_user_defined is not None and scene_user_defined.has_value(
            "scene_filter_file"
        ):
            scene_filter_file = scene_user_defined.get("scene_filter_file")
            # construct the dataset level path for the filter data file
            scene_filter_file = os.path.join(
                os.path.dirname(mm.active_dataset), scene_filter_file
            )
            with open(scene_filter_file, "r") as f:
                filter_json = json.load(f)
                for filter_type in [
                    "manually_filtered",
                    "access_filtered",
                    "stability_filtered",
                    "height_filtered",
                ]:
                    for filtered_unique_name in filter_json[filter_type]:
                        filtered_unique_names.append(filtered_unique_name)
            # add exclusion filters to all receptacles sets
            for _, r_set in self._receptacle_sets.items():
                r_set.excluded_receptacle_substrings.extend(
                    filtered_unique_names
                )
            logger.debug(
                f"Loaded receptacle filter data for scene '{scene_handle}' from configured filter file '{scene_filter_file}'."
            )

    def inc_count(self, recep_name: str) -> None:
        """
        Increment allowed objects for a Receptacle.
        :param recep_name: The unique name of the Receptacle.
        """
        if recep_name in self._receptacle_counts:
            self._receptacle_counts[recep_name] += 1

    def allocate_one_placement(self, allocated_receptacle: Receptacle) -> bool:
        """
        Record that a Receptacle has been allocated for one new object placement.
        If the Receptacle has a configured maximum number of remaining object placements, decrement that counter.
        If the Receptacle has no remaining allocations after this one, remove it from any existing ReceptacleSets to prevent it being sampled in the future.

        :param new_receptacle: The Receptacle with a new allocated object placement.

        :return: Whether or not the Receptacle has run out of remaining allocations.
        """
        recep_name = allocated_receptacle.unique_name
        if recep_name not in self._receptacle_counts:
            return False
        # decrement remaining allocations
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
