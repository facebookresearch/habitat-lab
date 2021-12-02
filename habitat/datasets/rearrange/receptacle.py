#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import magnum as mn
import numpy as np

import habitat_sim


class Receptacle:
    """
    Stores parameters necessary to define an AABB Receptacle for object sampling.
    """

    def __init__(
        self,
        name: str,
        bounds: mn.Range3D,
        rotation: Optional[mn.Quaternion] = None,
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
        nonzero_indices = np.nonzero(self.up)
        assert (
            len(nonzero_indices) == 1
        ), "The 'up' vector must be aligned with a primary axis for an AABB."
        self.up_axis = nonzero_indices[0]
        self.parent_object_handle = parent_object_handle
        self.is_parent_object_articulated = is_parent_object_articulated
        self.parent_link = parent_link
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
        Isolates boilerplate necessary to extract receptacle global transform from ROs and AOs.
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

    def sample_uniform_global(
        self, sim: habitat_sim.Simulator, sample_region_ratio: float
    ) -> mn.Vector3:
        """
        Sample a uniform random point in the local AABB and then transform it into global space.
        """
        local_sample = self.sample_uniform_local(sample_region_ratio)
        return self.get_global_transform(sim).transform_point(local_sample)


def get_all_scenedataset_receptacles(sim) -> Dict[str, Dict[str, List[str]]]:
    """
    Scrapes the active SceneDataset from a Simulator for all receptacles defined in rigid and articulated object templates.
    TODO: Note this will not include scene-specific overwrites, only receptacles included in object_config.json and ao_config.json files.
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


def find_receptacles(sim) -> List[Receptacle]:
    """
    Return a list of all receptacles scraped from the scene's currently instanced objects.
    """
    # TODO: Receptacles should be screened if the orientation will not support placement.
    obj_mgr = sim.get_rigid_object_manager()
    ao_mgr = sim.get_articulated_object_manager()

    receptacles: List[Receptacle] = []

    # search for global receptacles included with the stage
    stage_config = sim.get_stage_initialization_template()
    if stage_config is not None:
        stage_user_attr = stage_config.get_user_config()
        for sub_config_key in stage_user_attr.get_subconfig_keys():
            if sub_config_key.startswith("receptacle_"):
                sub_config = stage_user_attr.get_subconfig(sub_config_key)
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
                rotation = sub_config.get("rotation")

                receptacles.append(
                    Receptacle(
                        name=receptacle_name,
                        bounds=mn.Range3D.from_center(
                            sub_config.get("position"),
                            sub_config.get("scale"),
                        ),
                        rotation=rotation,
                        up=up,
                    )
                )

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
                        name=receptacle_name,
                        bounds=mn.Range3D.from_center(
                            sub_config.get("position"),
                            sub_config.get("scale"),
                        ),
                        up=up,
                        parent_object_handle=obj_handle,
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
                        name=receptacle_name,
                        bounds=mn.Range3D.from_center(
                            sub_config.get("position") * obj.global_scale,
                            sub_config.get("scale") * obj.global_scale,
                        ),
                        up=up,
                        parent_object_handle=obj_handle,
                        is_parent_object_articulated=True,
                        parent_link=parent_link_ix,
                    )
                )

    return receptacles
