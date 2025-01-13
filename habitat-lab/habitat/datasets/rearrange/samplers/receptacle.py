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
from typing import Any, Dict, List, Optional, Union

import corrade as cr
import magnum as mn
import numpy as np

import habitat.sims.habitat_simulator.sim_utilities as sutils
import habitat_sim
from habitat.core.logging import logger
from habitat.datasets.rearrange.navmesh_utils import is_accessible
from habitat.sims.habitat_simulator.debug_visualizer import dblr_draw_bb
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
        :param parent_link: Index of the link to which the Receptacle is attached if the parent is an ArticulatedObject. -1 denotes the base link. None for rigid objects and stage Receptacles.
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
    def bounds(self) -> mn.Range3D:
        """
        AABB of the Receptacle in local space.
        Default is empty Range3D.
        """
        return mn.Range3D()

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

    def sample_uniform_global(
        self, sim: habitat_sim.Simulator, sample_region_scale: float
    ) -> mn.Vector3:
        """
        Sample a uniform random point in the local Receptacle volume and then transform it into global space.

        :param sample_region_scale: defines a XZ scaling of the sample region around its center.
        """
        local_sample = self.sample_uniform_local(sample_region_scale)
        return self.get_global_transform(sim).transform_point(local_sample)

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

    def get_support_object_ids(self, sim: habitat_sim.Simulator) -> List[int]:
        """
        Get a list of object ids representing the set of acceptable support surfaces for this receptacle.

        :param sim: The Simulator instance.
        :return: A list of object id integers for this Receptacle's set of valid support surfaces.
        """
        if self.parent_object_handle is None:
            # this is the stage
            return [habitat_sim.stage_id]

        parent_object = sutils.get_obj_from_handle(
            sim, self.parent_object_handle
        )
        if parent_object.is_articulated:
            if self.parent_link <= 0:
                # Receptacle is attached to the body link, so only allow placements there
                # NOTE: If collision objects are marked STATIC in the URDF (via collision_group==2) then they will be attached to the -1 link as STATIC rigids, even if defined at the 0 link
                return [
                    parent_object.object_id,
                    parent_object.link_ids_to_object_ids[0],
                ]
            else:
                # Receptacle is attached to a moveable link, only allow samples on that link
                return [parent_object.link_ids_to_object_ids[self.parent_link]]

        # for rigid objects support surface is the object_id
        return [parent_object.object_id]

    def dist_to_rec(
        self, sim: habitat_sim.Simulator, point: np.ndarray
    ) -> float:
        """
        Compute and return the distance from a 3D global point to the Receptacle.

        :param sim: The Simulator instance for querying global transforms.
        :param point: A 3D point in global space. E.g. the bottom center point of a placed object.
        :return: Point to Receptacle distance.
        """
        raise NotImplementedError


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
        :param parent_link: Index of the link to which the Receptacle is attached if the parent is an ArticulatedObject. -1 denotes the base link. None for rigid objects and stage Receptacles.
        :param rotation: Optional rotation of the Receptacle AABB. Only used for globally defined stage Receptacles to provide flexibility.
        """
        super().__init__(name, parent_object_handle, parent_link, up)
        self._bounds = bounds
        self.rotation = rotation if rotation is not None else mn.Quaternion()

    @property
    def bounds(self) -> mn.Range3D:
        """
        AABB of the Receptacle in local space.
        """
        return self._bounds

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
            # TODO: add an API query or other method to avoid reconstructing the stage frame here
            stage_config = sim.get_stage_initialization_template()

            r_frameup_worldup = mn.Quaternion.rotation(
                habitat_sim.geo.UP, stage_config.orient_up.normalized()
            )
            v_prime = r_frameup_worldup.transform_vector_normalized(
                habitat_sim.geo.FRONT
            ).normalized()
            world_to_local = (
                mn.Quaternion.rotation(
                    v_prime, stage_config.orient_front.normalized()
                )
                * r_frameup_worldup
            ).normalized()

            local_to_world = world_to_local.inverted()
            l2w4 = mn.Matrix4.from_(local_to_world.to_matrix(), mn.Vector3())

            # apply the receptacle rotation from the bb center
            T = mn.Matrix4.from_(mn.Matrix3(), self.bounds.center())
            R = mn.Matrix4.from_(self.rotation.to_matrix(), mn.Vector3())
            # translate frame to center, rotate, translate back
            l2w4 = l2w4.__matmul__(T.__matmul__(R).__matmul__(T.inverted()))
            return l2w4

        # base class implements getting transform from attached objects
        return super().get_global_transform(sim)

    def debug_draw(
        self, sim: habitat_sim.Simulator, color: Optional[mn.Color4] = None
    ) -> None:
        """
        Render the AABBReceptacle with DebugLineRender utility at the current frame.
        Must be called after each frame is rendered, before querying the image data.

        :param sim: Simulator must be provided.
        :param color: Optionally provide wireframe color, otherwise magenta.
        """
        dblr_draw_bb(
            sim.get_debug_line_render(),
            self.bounds,
            self.get_global_transform(sim),
            color,
        )


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
        scale: Union[float, mn.Vector3] = None,
    ) -> None:
        """
        Initialize the TriangleMeshReceptacle from mesh data and pre-compute the area weighted accumulator.

        :param name: The name of the Receptacle. Should be unique and descriptive for any one object.
        :param mesh_data: The Receptacle's mesh data. A magnum.trade.MeshData object (indices len divisible by 3).
        :param parent_object_handle: The rigid or articulated object instance handle for the parent object to which the Receptacle is attached. None for globally defined stage Receptacles.
        :param parent_link: Index of the link to which the Receptacle is attached if the parent is an ArticulatedObject. -1 denotes the base link. None for rigid objects and stage Receptacles.
        :param up: The "up" direction of the Receptacle in local AABB space. Used for optionally culling receptacles in un-supportive states such as inverted surfaces.
        :param scale: The scaling vector (or uniform scaling float) to be applied to the mesh.
        """
        super().__init__(name, parent_object_handle, parent_link, up)
        self.mesh_data = mesh_data

        # apply the scale
        if scale is not None:
            m_verts = self.mesh_data.mutable_attribute(
                mn.trade.MeshAttribute.POSITION
            )
            for vix, v in enumerate(m_verts):
                m_verts[vix] = v * scale

        self.area_weighted_accumulator = (
            []
        )  # normalized float weights for each triangle for sampling
        assert_triangles(mesh_data.indices)

        # pre-compute the normalized cumulative area of all triangle faces for later sampling
        self.total_area = 0.0
        self.triangles = []
        for f_ix in range(int(len(mesh_data.indices) / 3)):
            v = self.get_face_verts(f_ix)
            w1 = v[1] - v[0]
            w2 = v[2] - v[1]
            self.triangles.append(v)
            self.area_weighted_accumulator.append(
                0.5 * mn.math.cross(w1, w2).length()
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
        minv = mn.Vector3(mn.math.inf)
        maxv = mn.Vector3(-mn.math.inf)
        for v in self.mesh_data.attribute(mn.trade.MeshAttribute.POSITION):
            minv = mn.math.min(minv, v)
            maxv = mn.math.max(maxv, v)
        minmax = (minv, maxv)
        self._bounds = mn.Range3D(minmax)

    @property
    def bounds(self) -> mn.Range3D:
        """
        Get the vertex AABB bounds pre-computed during initialization.
        """
        return self._bounds

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
        for verts in self.triangles:
            for edge in range(3):
                dblr.draw_transformed_line(
                    verts[edge], verts[(edge + 1) % 3], color
                )
        dblr.pop_transform()

    def dist_to_rec(
        self, sim: habitat_sim.Simulator, point: np.ndarray
    ) -> float:
        """
        Compute and return the distance from a 3D global point to the Receptacle. Uses point to mesh distance check.

        :param sim: The Simulator instance.
        :param point: A 3D point in global space. E.g. the bottom center point of a placed object.
        :return: Point to Receptacle distance.
        """
        t_form = self.get_global_transform(sim)
        # optimization: transform the point into local space instead of transforming the mesh into global space
        local_point = t_form.inverted().transform_point(point)
        # iterate over the triangles, getting point to edge distances
        # NOTE: list of lists, each with 3 numpy arrays, one for each vertex
        np_tri = np.array(self.triangles)
        np_point = np.array(local_point)
        # compute the minimum point to mesh distance
        p_to_t_dist = sutils.point_to_tri_dist(np_point, np_tri)[0]
        return p_to_t_dist


class AnyObjectReceptacle(Receptacle):
    """
    The AnyObjectReceptacle enables any rigid or articulated object or link to be used as a Receptacle without metadata annotation.
    It uses the top surface of an object's global space bounding box as a heuristic for the sampling area.
    The sample efficiency is likely to be poor (especially for concave objects like L-shaped sofas), TODO: this could be mitigated by the option to pre-compute a discrete set of candidate points via raycast upon initialization.
    Also, this heuristic will not support use of interior surfaces such as cubby and cabinet shelves since volumetric occupancy is not considered.

    Note the caveats above and consider that the ideal application of the AnyObjectReceptacle is to support placement of objects onto other simple objects such as open face crates, bins, baskets, trays, plates, bowls, etc... for which receptacle annotation would be overkill.
    """

    def __init__(
        self,
        name: str,
        parent_object_handle: str = None,
        parent_link: Optional[int] = None,
    ):
        """
        Initialize the object as a Receptacle.

        :param precompute_candidate_pointset: Whether or not to pre-compute and cache a discrete point set for sampling instead of using the global bounding box. Uses raycasting with rejection sampling.
        """

        super().__init__(name, parent_object_handle, parent_link)

    def _get_global_bb(self, sim: habitat_sim.Simulator) -> mn.Range3D:
        """
        Get the global AABB of the Receptacle parent object.
        """

        obj = sutils.get_obj_from_handle(sim, self.parent_object_handle)

        # get the global keypoints of the object
        receptacle_bb, local_to_global = None, None
        if self.parent_link is not None and self.parent_link >= 0:
            link_node = obj.get_link_scene_node(self.parent_link)
            receptacle_bb = link_node.cumulative_bb
            local_to_global = link_node.absolute_transformation()
        else:
            receptacle_bb = obj.aabb
            local_to_global = obj.transformation
        global_keypoints = sutils.get_global_keypoints_from_bb(
            receptacle_bb, local_to_global
        )

        # find min and max
        global_bb = mn.Range3D(
            np.min(global_keypoints, axis=0), np.max(global_keypoints, axis=0)
        )

        return global_bb

    @property
    def bounds(self) -> mn.Range3D:
        """
        AABB of the Receptacle in local space.
        NOTE: this is an effortful query, not a getter.
        TODO: This needs a sim instance to compute the global bounding box
        """

        # TODO: grab the bounds from the global AABB at this state?
        # return mn.Range3D()
        raise NotImplementedError

    def sample_uniform_local(
        self, sample_region_scale: float = 1.0
    ) -> mn.Vector3:
        """
        Sample a uniform random point within Receptacle in local space.
        NOTE: This only works if a pointset cache was pre-computed. Otherwise raises an exception.

        :param sample_region_scale: defines a XZ scaling of the sample region around its center. For example to constrain object spawning toward the center of a receptacle.
        """

        raise NotImplementedError

    def sample_uniform_global(
        self, sim: habitat_sim.Simulator, sample_region_scale: float
    ) -> mn.Vector3:
        """
        Sample a uniform random point on the top surface of the global bounding box of the object.
        TODO: If a pre-computed candidate point set was cached, simply sample from those points instead.

        :param sample_region_scale: defines a XZ scaling of the sample region around its center. No-op for cached points.
        """

        aabb = self._get_global_bb(sim)
        if sample_region_scale != 1.0:
            aabb = mn.Range3D.from_center(
                aabb.center(),
                aabb.scaled(
                    mn.Vector3d(sample_region_scale, 1, sample_region_scale)
                ).size()
                / 2.0,
            )

        sample = np.random.uniform(aabb.back_top_left, aabb.front_top_right)
        return sample

    def debug_draw(
        self, sim: habitat_sim.Simulator, color: Optional[mn.Color4] = None
    ) -> None:
        """
        Render the Receptacle with DebugLineRender utility at the current frame.
        Must be called after each frame is rendered, before querying the image data.

        :param sim: Simulator must be provided.
        :param color: Optionally provide wireframe color, otherwise magenta.
        """

        aabb = self._get_global_bb(sim)
        top_min = aabb.min
        top_min[1] = aabb.top
        top_max = aabb.max
        top_max[1] = aabb.top
        top_range = mn.Range3D(top_min, top_max)
        dblr_draw_bb(sim.get_debug_line_render(), top_range, color=color)


def get_all_scenedataset_receptacles(
    sim: habitat_sim.Simulator,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Scrapes the active SceneDataset from a Simulator for all receptacle names defined in rigid/articulated object and stage templates for investigation and preview purposes.
    Note this will not include scene-specific overrides defined in scene_config.json files. Only receptacles defined in object_config.json, ao_config.json, and stage_config.json files or added programmatically to associated Attributes objects will be found.

    Returns a dict with keys {"stage", "rigid", "articulated"} mapping object template handles to lists of receptacle names.

    :param sim: Simulator must be provided.
    """
    # cache the rigid and articulated receptacles separately
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
                if template_handle not in receptacles["stage"]:
                    receptacles["stage"][template_handle] = []
                receptacles["stage"][template_handle].append(item)

    # scrape the rigid object configs:
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


def filter_interleave_mesh(mesh: mn.trade.MeshData) -> mn.trade.MeshData:
    """
    Filter all but position data and interleave a mesh to reduce overall memory footprint.
    Convert triangle like primitives into triangles and assert only triangles remain.

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
    mesh = mn.meshtools.interleave(
        mesh, flags=mn.meshtools.InterleaveFlags.NONE
    )

    return mesh


def import_tri_mesh(mesh_file: str) -> List[mn.trade.MeshData]:
    """
    Returns a list of MeshData objects from a mesh asset using magnum trade importer.

    :param mesh_file: The input meshes file. NOTE: must contain only triangles.
    """
    _manager.set_preferred_plugins("StanfordImporter", ["AssimpImporter"])
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
    :param parent_template_directory: The filesystem directory path containing the configuration file. Used to construct the absolute asset path from the relative asset path.
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
                    parent_template_directory,
                    sub_config.get("mesh_filepath"),
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
                            scale=ao_uniform_scaling,
                        )
                    )
            else:
                raise AssertionError(
                    f"Receptacle detected without a subtype specifier: '{mesh_receptacle_id_string}'"
                )

    return receptacles


def cull_filtered_receptacles(
    receptacles: List[Receptacle], exclude_filter_strings: List[str]
) -> List[Receptacle]:
    """
    Filter a list of Receptacles to exclude any which are matched to the provided exclude_filter_strings.
    Each string in filter strings is checked against each receptacle's unique_name. If the unique_name contains any filter string as a substring, that Receptacle is filtered.

    :param receptacles: The initial list of Receptacle objects.
    :param exclude_filter_strings: The list of filter substrings defining receptacles which should not be active in the current scene.

    :return: The filtered list of Receptacle objects. Those which contain none of the filter substrings in their unqiue_name.
    """

    filtered_receptacles = []
    for receptacle in receptacles:
        culled = False
        for filter_substring in exclude_filter_strings:
            if filter_substring in receptacle.unique_name:
                culled = True
                break
        if not culled:
            filtered_receptacles.append(receptacle)
    return filtered_receptacles


def find_receptacles(
    sim: habitat_sim.Simulator,
    ignore_handles: Optional[List[str]] = None,
    exclude_filter_strings: Optional[List[str]] = None,
) -> List[Union[Receptacle, AABBReceptacle, TriangleMeshReceptacle]]:
    """
    Scrape and return a list of all Receptacles defined in the metadata belonging to the scene's currently instanced objects.

    :param sim: Simulator must be provided.
    :param ignore_handles: An optional list of handles for ManagedObjects which should be skipped. No Receptacles for matching objects will be returned.
    :param exclude_filter_strings: An optional list of excluded Receptacle substrings. Any Receptacle which contains any excluded filter substring in its unique_name will not be included in the returned set.
    """

    obj_mgr = sim.get_rigid_object_manager()
    ao_mgr = sim.get_articulated_object_manager()
    if ignore_handles is None:
        ignore_handles = []

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
        if obj_handle in ignore_handles:
            continue
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
        if obj_handle in ignore_handles:
            continue
        obj = ao_mgr.get_object_by_handle(obj_handle)
        # TODO: no way to get filepath from AO currently. Add this API.
        source_template_file = ""
        creation_attr = obj.creation_attributes
        if creation_attr is not None:
            source_template_file = creation_attr.file_directory
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

    # filter out individual Receptacles with excluded substrings
    if exclude_filter_strings is not None:
        receptacles = cull_filtered_receptacles(
            receptacles, exclude_filter_strings
        )

    # check for non-unique naming mistakes in user dataset
    for rec_ix in range(len(receptacles)):
        rec1_unique_name = receptacles[rec_ix].unique_name
        for rec_ix2 in range(rec_ix + 1, len(receptacles)):
            assert (
                rec1_unique_name != receptacles[rec_ix2].unique_name
            ), "Two Receptacles found with the same unique name '{rec1_unique_name}'. Likely indicates multiple receptacle entries with the same name in the same config."

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


def get_scene_rec_filter_filepath(
    mm: habitat_sim.metadata.MetadataMediator, scene_handle: str
) -> str:
    """
    Look in the user_defined metadata for a scene to find the configured filepath for the scene's Receptacle filter file.

    :return: Filter filepath or None if not found.
    """
    scene_user_defined = mm.get_scene_user_defined(scene_handle)
    if scene_user_defined is not None and scene_user_defined.has_value(
        "scene_filter_file"
    ):
        scene_filter_file = scene_user_defined.get("scene_filter_file")
        scene_filter_file = os.path.join(
            os.path.dirname(mm.active_dataset), scene_filter_file
        )
        return scene_filter_file
    return None


def get_excluded_recs_from_filter_file(
    rec_filter_filepath: str, filter_types: Optional[List[str]] = None
) -> List[str]:
    """
    Load and digest a Receptacle filter file to generate a list of Receptacle.unique_names strings which should be excluded from the active ReceptacleSet.

    :param filter_types: Optionally specify a particular set of filter types to scrape. Default is all exclusion filters.
    """

    possible_exclude_filter_types = [
        "manually_filtered",
        "access_filtered",
        "stability_filtered",
        "height_filtered",
    ]

    if filter_types is None:
        filter_types = possible_exclude_filter_types
    else:
        for filter_type in filter_types:
            assert (
                filter_type in possible_exclude_filter_types
            ), f"Specified filter type '{filter_type}' is not in supported set: {possible_exclude_filter_types}"

    return get_recs_from_filter_file(rec_filter_filepath, filter_types)


def get_recs_from_filter_file(
    rec_filter_filepath: str, filter_types: List[str]
) -> List[str]:
    """
    Load and digest a Receptacle filter file to generate a list of Receptacle.unique_names which belong to a particular filter subset.

    :param filter_types: Specify a particular subset of filter types to include.
    """

    # all allowed filter set types include:
    all_possible_filter_types = [
        "active",
        "manually_filtered",
        "access_filtered",
        "stability_filtered",
        "height_filtered",
        "within_set",
    ]

    # check that specified query filter types are valid
    for filter_type in filter_types:
        assert (
            filter_type in all_possible_filter_types
        ), f"Specified filter type '{filter_type}' is not in supported set: {all_possible_filter_types}"

    filtered_unique_names = []
    with open(rec_filter_filepath, "r") as f:
        filter_json = json.load(f)
        for filter_type in filter_types:
            if filter_type in filter_json:
                for filtered_unique_name in filter_json[filter_type]:
                    filtered_unique_names.append(filtered_unique_name)
            else:
                logger.warning(
                    f"The filter file '{rec_filter_filepath}' does not contain the requested filter type '{filter_type}'."
                )
    return list(set(filtered_unique_names))


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
        scene_filter_filepath = get_scene_rec_filter_filepath(mm, scene_handle)
        if scene_filter_filepath is not None:
            filtered_unique_names = get_excluded_recs_from_filter_file(
                scene_filter_filepath
            )
            # add exclusion filters to all receptacles sets
            for r_set in self._receptacle_sets.values():
                r_set.excluded_receptacle_substrings.extend(
                    filtered_unique_names
                )
                logger.info(
                    f"Loaded receptacle filter data for scene '{scene_handle}' from configured filter file '{scene_filter_filepath}'."
                )
        else:
            logger.info(
                f"Loaded receptacle filter data for scene '{scene_handle}' does not have configured filter file."
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


def get_obj_manager_for_receptacle(
    sim: habitat_sim.Simulator, receptacle: Receptacle
) -> Union[
    habitat_sim.physics.RigidObjectManager,
    habitat_sim.physics.ArticulatedObjectManager,
]:
    """
    Get the correct object manager for the Receptacle.

    :param sim: The Simulator instance.
    :param receptacle: The Receptacle instance.

    :return: Either RigidObjectManager or ArticulatedObjectManager.
    """
    if receptacle.is_parent_object_articulated:
        obj_mgr = sim.get_articulated_object_manager()
    else:
        obj_mgr = sim.get_rigid_object_manager()
    return obj_mgr


def get_navigable_receptacles(
    sim: habitat_sim.Simulator,
    receptacles: List[Receptacle],
    nav_island: int,
    nav_to_min_distance: float = 1.5,
) -> List[Receptacle]:
    """
    Given a list of receptacles, return the ones that are heuristically navigable from the largest indoor navmesh island.

    Navigability heuristic is that at least two Receptacle AABB corners are within 1.5m of the largest indoor navmesh island and object is within 0.2m of the configured agent height.

    :param sim: The Simulator instance.
    :param receptacles: The list of Receptacle instances to cull.
    :param nav_island: The NavMesh island on which to check accessibility. -1 is the full NavMesh.
    :param nav_to_min_distance: Minimum distance threshold. -1 opts out of the test and returns True (i.e. no minimum distance).

    :return: The list of heuristic passing Receptacle instances.
    """
    # The receptacle should be unoccluded from this height
    max_access_height = 1.3
    navigable_receptacles: List[Receptacle] = []
    for receptacle in receptacles:
        receptacle_obj = sutils.get_obj_from_handle(
            sim, receptacle.parent_object_handle
        )

        # get the global bounding box of the object
        receptacle_bb = None
        if receptacle.parent_link >= 0:
            link_node = receptacle_obj.get_link_scene_node(
                receptacle.parent_link
            )
            receptacle_bb = habitat_sim.geo.get_transformed_bb(
                link_node.cumulative_bb, link_node.absolute_transformation()
            )
        else:
            receptacle_bb = habitat_sim.geo.get_transformed_bb(
                receptacle_obj.aabb, receptacle_obj.transformation
            )

        recep_points = [
            receptacle_bb.back_bottom_left,
            receptacle_bb.back_bottom_right,
            receptacle_bb.front_bottom_left,
            receptacle_bb.front_bottom_right,
        ]
        # At least 2 corners should be accessible
        corners_accessible = True
        corners_accessible = (
            sum(
                is_accessible(
                    sim=sim,
                    point=point,
                    height=max_access_height,
                    nav_to_min_distance=nav_to_min_distance,
                    nav_island=nav_island,
                    target_object_ids=[receptacle_obj.object_id],
                )
                for point in recep_points
            )
            >= 2
        )

        if not corners_accessible:
            logger.info(
                f"Receptacle {receptacle.parent_object_handle}, {receptacle_obj.translation} is not accessible."
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
