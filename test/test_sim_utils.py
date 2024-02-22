#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp

import magnum as mn
import pytest

from habitat.sims.habitat_simulator.sim_utilities import (
    above,
    bb_ray_prescreen,
    get_all_object_ids,
    get_all_objects,
    get_ao_link_id_map,
    get_obj_from_handle,
    get_obj_from_id,
    object_keypoint_cast,
    snap_down,
)
from habitat_sim import Simulator, built_with_bullet
from habitat_sim.metadata import MetadataMediator
from habitat_sim.physics import MotionType
from habitat_sim.utils.settings import default_sim_settings, make_cfg


@pytest.mark.skipif(
    not built_with_bullet,
    reason="ArticulatedObject API requires Bullet physics.",
)
@pytest.mark.skipif(
    not osp.exists("data/test_assets/scenes/plane.glb"),
    reason="Requires the plane.glb habitat test asset",
)
@pytest.mark.parametrize(
    "support_margin",
    [0.0, 0.04, 0.1],
)
@pytest.mark.parametrize("obj_margin", [0.0, 0.04, 0.1])
@pytest.mark.parametrize("stage_support", [True, False])
def test_snap_down(support_margin, obj_margin, stage_support):
    """
    Test snapping objects onto stages and other assets.
    """

    mm = MetadataMediator()

    otm = mm.object_template_manager
    stm = mm.stage_template_manager

    # prepare the support object depending on 'stage_support' mode. Either a STATIC object or a stage mesh.
    cube_template_handle = otm.get_template_handles("cubeSolid")[0]
    cube_stage_template_handle = "cube_stage_object"
    plane_stage_template_handle = "plane_stage"
    if not stage_support:
        # setup a cube ground plane object config
        cube_template = otm.get_template_by_handle(cube_template_handle)
        cube_template.scale = mn.Vector3(10, 0.05, 10)
        cube_template.margin = support_margin
        otm.register_template(cube_template, cube_stage_template_handle)
    else:
        # setup a stage using the plane.glb test asset
        new_stage_template = stm.create_new_template(
            handle=plane_stage_template_handle
        )
        new_stage_template.render_asset_handle = (
            "data/test_assets/scenes/plane.glb"
        )
        new_stage_template.margin = support_margin
        new_stage_template.orient_up = mn.Vector3(0, 0, 1)
        new_stage_template.orient_front = mn.Vector3(0, 1, 0)
        # need to make the scale reasonable or navmesh takes forever to recompute
        # BUG: this scale is not used by sim currently...
        new_stage_template.scale = mn.Vector3(0.01, 1.0, 0.01)
        # temporary hack: load and arbitrary navmesh, we don't use it anyway
        new_stage_template.navmesh_asset_handle = (
            "data/test_assets/scenes/simple_room.stage_config.navmesh"
        )
        stm.register_template(
            template=new_stage_template,
            specified_handle=plane_stage_template_handle,
        )

    # setup test cube object config
    cube_template = otm.get_template_by_handle(cube_template_handle)
    cube_template.margin = obj_margin
    otm.register_template(cube_template)

    # Test snapping a cube object onto another object
    sim_settings = default_sim_settings.copy()
    sim_settings["sensor_height"] = 0.0
    sim_settings["scene"] = "NONE"
    if stage_support:
        sim_settings["scene"] = plane_stage_template_handle
    hab_cfg = make_cfg(sim_settings)
    hab_cfg.metadata_mediator = mm
    with Simulator(hab_cfg) as sim:
        rom = sim.get_rigid_object_manager()

        # add the cube objects
        cube_stage_obj = None
        support_obj_ids = [-1]
        if not stage_support:
            cube_stage_obj = rom.add_object_by_template_handle(
                cube_stage_template_handle
            )
            assert (
                cube_stage_obj.is_alive
            ), "Failure to add object may indicate configuration issue or no 'cube_stage_template_handle'."
            support_obj_ids = [cube_stage_obj.object_id]
        cube_obj = rom.add_object_by_template_handle(cube_template_handle)
        assert cube_obj.is_alive

        # test with various combinations of motion type for both objects
        for object_motion_type in [MotionType.KINEMATIC, MotionType.DYNAMIC]:
            for support_motion_type in [
                MotionType.STATIC,
                MotionType.KINEMATIC,
                MotionType.DYNAMIC,
            ]:
                if not stage_support:
                    cube_stage_obj.motion_type = support_motion_type
                cube_obj.motion_type = object_motion_type

                # snap will fail because object COM is inside the support surface shape so raycast won't detect the support surface
                initial_translation = mn.Vector3(0, 0, 0.1)
                cube_obj.translation = initial_translation
                snap_success = snap_down(
                    sim, cube_obj, support_obj_ids=support_obj_ids
                )
                assert not snap_success
                assert (
                    initial_translation - cube_obj.translation
                ).length() < 1e-5, (
                    "Translation should not be changed after snap failure."
                )
                bb_ray_prescreen_results = bb_ray_prescreen(
                    sim, cube_obj, support_obj_ids=support_obj_ids
                )
                assert bb_ray_prescreen_results["surface_snap_point"] is None

                # with object above the support, snap will succeed.
                cube_obj.translation = mn.Vector3(0, 0.2, 0)
                snap_success = snap_down(
                    sim, cube_obj, support_obj_ids=support_obj_ids
                )
                assert snap_success
                bb_ray_prescreen_results = bb_ray_prescreen(
                    sim, cube_obj, support_obj_ids=support_obj_ids
                )
                assert (
                    cube_obj.translation
                    - bb_ray_prescreen_results["surface_snap_point"]
                ).length() < 1e-5, (
                    "Translation should be the pre-screened location."
                )
                assert (
                    bb_ray_prescreen_results["surface_snap_point"] is not None
                )
                if stage_support:
                    # don't need 3 iterations for stage b/c no motion types to test
                    break


@pytest.mark.skipif(
    not built_with_bullet,
    reason="Raycasting API requires Bullet physics.",
)
@pytest.mark.skipif(
    not osp.exists("data/replica_cad/"),
    reason="Requires ReplicaCAD dataset.",
)
def test_object_getters():
    sim_settings = default_sim_settings.copy()
    sim_settings[
        "scene_dataset_config_file"
    ] = "data/replica_cad/replicaCAD.scene_dataset_config.json"
    sim_settings["scene"] = "apt_0"
    hab_cfg = make_cfg(sim_settings)
    with Simulator(hab_cfg) as sim:
        # scrape various lists from utils
        all_objects = get_all_objects(sim)
        all_object_ids = get_all_object_ids(sim)
        ao_link_map = get_ao_link_id_map(sim)

        # validate parity between util results
        assert len(all_objects) == (
            sim.get_rigid_object_manager().get_num_objects()
            + sim.get_articulated_object_manager().get_num_objects()
        )
        for object_id in ao_link_map:
            assert (
                object_id in all_object_ids
            ), f"Link or AO object id {object_id} is not found in the global object id map."
        for obj in all_objects:
            assert obj is not None
            assert obj.is_alive
            assert (
                obj.object_id in all_object_ids
            ), f"Object's object_id {object_id} is not found in the global object id map."
            # check the wrapper getter functions
            obj_from_id_getter = get_obj_from_id(
                sim, obj.object_id, ao_link_map
            )
            obj_from_handle_getter = get_obj_from_handle(sim, obj.handle)
            assert obj_from_id_getter.object_id == obj.object_id
            assert obj_from_handle_getter.object_id == obj.object_id

        # specifically validate link object_id mapping
        aom = sim.get_articulated_object_manager()
        for ao in aom.get_objects_by_handle_substring().values():
            assert ao.object_id in all_object_ids
            assert ao.object_id in ao_link_map
            link_indices = ao.get_link_ids()
            assert len(ao.link_object_ids) == len(link_indices)
            for link_object_id, link_index in ao.link_object_ids.items():
                assert link_object_id in ao_link_map
                assert ao_link_map[link_object_id] == ao.object_id
                assert link_index in link_indices
                # links should return reference to parent object
                obj_from_id_getter = get_obj_from_id(
                    sim, link_object_id, ao_link_map
                )
                assert obj_from_id_getter.object_id == ao.object_id


@pytest.mark.skipif(
    not built_with_bullet,
    reason="Raycasting API requires Bullet physics.",
)
@pytest.mark.skipif(
    not osp.exists("data/replica_cad/"),
    reason="Requires ReplicaCAD dataset.",
)
def test_keypoint_cast_prepositions():
    sim_settings = default_sim_settings.copy()
    sim_settings[
        "scene_dataset_config_file"
    ] = "data/replica_cad/replicaCAD.scene_dataset_config.json"
    sim_settings["scene"] = "apt_0"
    hab_cfg = make_cfg(sim_settings)
    with Simulator(hab_cfg) as sim:
        all_objects = get_all_object_ids(sim)

        mixer_object = get_obj_from_handle(
            sim, "frl_apartment_small_appliance_01_:0000"
        )
        mixer_above = above(sim, mixer_object)
        mixer_above_strings = [
            all_objects[obj_id] for obj_id in mixer_above if obj_id >= 0
        ]
        expected_mixer_above_strings = [
            "kitchen_counter_:0000",
            "kitchen_counter_:0000 -- drawer2_bottom",
            "kitchen_counter_:0000 -- drawer2_middle",
            "kitchen_counter_:0000 -- drawer2_top",
        ]
        for expected in expected_mixer_above_strings:
            assert expected in mixer_above_strings
        assert len(mixer_above_strings) == len(expected_mixer_above_strings)

        tv_object = get_obj_from_handle(sim, "frl_apartment_tv_screen_:0000")
        tv_above = above(sim, tv_object)
        tv_above_strings = [
            all_objects[obj_id] for obj_id in tv_above if obj_id >= 0
        ]
        expected_tv_above_strings = [
            "frl_apartment_tvstand_:0000",
            "frl_apartment_chair_01_:0000",
        ]

        for expected in expected_tv_above_strings:
            assert expected in tv_above_strings
        assert len(tv_above_strings) == len(expected_tv_above_strings)

        # now define a custom keypoint cast from the mixer constructed to include tv in the set
        mixer_to_tv = (
            tv_object.translation - mixer_object.translation
        ).normalized()
        mixer_to_tv_object_ids = [
            hit.object_id
            for keypoint_raycast_result in object_keypoint_cast(
                sim, mixer_object, direction=mixer_to_tv
            )
            for hit in keypoint_raycast_result.hits
        ]
        mixer_to_tv_object_ids = list(set(mixer_to_tv_object_ids))
        assert tv_object.object_id in mixer_to_tv_object_ids
