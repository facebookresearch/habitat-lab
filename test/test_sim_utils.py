#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp

import magnum as mn
import pytest

import habitat.sims.habitat_simulator.sim_utilities as sutils
from habitat_sim import Simulator, built_with_bullet, stage_id
from habitat_sim.metadata import MetadataMediator
from habitat_sim.physics import JointType, MotionType
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
        # stage defaults to ID specified as constant in habitat_sim.stage_id
        support_obj_ids = [stage_id]
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
                snap_success = sutils.snap_down(
                    sim, cube_obj, support_obj_ids=support_obj_ids
                )
                assert not snap_success
                assert (
                    initial_translation - cube_obj.translation
                ).length() < 1e-5, (
                    "Translation should not be changed after snap failure."
                )
                bb_ray_prescreen_results = sutils.bb_ray_prescreen(
                    sim, cube_obj, support_obj_ids=support_obj_ids
                )
                assert bb_ray_prescreen_results["surface_snap_point"] is None

                # with object above the support, snap will succeed.
                cube_obj.translation = mn.Vector3(0, 0.2, 0)
                snap_success = sutils.snap_down(
                    sim, cube_obj, support_obj_ids=support_obj_ids
                )
                assert snap_success
                bb_ray_prescreen_results = sutils.bb_ray_prescreen(
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

                # reset the object and try again, ignoring the supports instead
                cube_obj.translation = mn.Vector3(0, 0.2, 0)
                snap_success = sutils.snap_down(
                    sim, cube_obj, ignore_obj_ids=support_obj_ids
                )
                assert not snap_success

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
        all_objects = sutils.get_all_objects(sim)
        all_object_ids = sutils.get_all_object_ids(sim)
        ao_link_map = sutils.get_ao_link_id_map(sim)

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
            obj_from_id_getter = sutils.get_obj_from_id(
                sim, obj.object_id, ao_link_map
            )
            obj_from_handle_getter = sutils.get_obj_from_handle(
                sim, obj.handle
            )
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
                obj_from_id_getter = sutils.get_obj_from_id(
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
        all_objects = sutils.get_all_object_ids(sim)

        mixer_object = sutils.get_obj_from_handle(
            sim, "frl_apartment_small_appliance_01_:0000"
        )
        mixer_above = sutils.above(sim, mixer_object)
        mixer_above_strings = [
            all_objects[obj_id] for obj_id in mixer_above if obj_id > stage_id
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

        tv_object = sutils.get_obj_from_handle(
            sim, "frl_apartment_tv_screen_:0000"
        )
        tv_above = sutils.above(sim, tv_object)
        tv_above_strings = [
            all_objects[obj_id] for obj_id in tv_above if obj_id > stage_id
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
            for keypoint_raycast_result in sutils.object_keypoint_cast(
                sim, mixer_object, direction=mixer_to_tv
            )
            for hit in keypoint_raycast_result.hits
        ]
        mixer_to_tv_object_ids = list(set(mixer_to_tv_object_ids))
        assert tv_object.object_id in mixer_to_tv_object_ids

        # now test "within" preposition

        # the clock is sitting within the shelf object
        clock_obj = sutils.get_obj_from_handle(
            sim, "frl_apartment_clock_:0000"
        )
        shelf_object = sutils.get_obj_from_handle(
            sim, "frl_apartment_wall_cabinet_01_:0000"
        )
        clock_within = sutils.within(sim, clock_obj)
        assert shelf_object.object_id in clock_within
        assert len(clock_within) == 1

        # now check borderline containment of a canister object in a basket
        canister_object = sutils.get_obj_from_handle(
            sim, "frl_apartment_kitchen_utensil_08_:0000"
        )
        basket_object = sutils.get_obj_from_handle(
            sim, "frl_apartment_basket_:0000"
        )

        # place the canister just above, but outside the basket
        canister_object.translation = mn.Vector3(-2.01639, 1.35, 0.0410867)
        canister_within = sutils.within(sim, canister_object)
        assert len(canister_within) == 0

        # move it slightly downward such that the extremal keypoints are contained.
        canister_object.translation = mn.Vector3(-2.01639, 1.3, 0.0410867)
        canister_within = sutils.within(sim, canister_object)
        assert len(canister_within) == 1
        assert basket_object.object_id in canister_within
        # now make the check more strict, requiring 6 keypoints
        canister_within = sutils.within(
            sim, canister_object, keypoint_vote_threshold=6
        )
        assert len(canister_within) == 0

        # further lower the canister such that the center is contained
        canister_object.translation = mn.Vector3(-2.01639, 1.2, 0.0410867)
        # when center ensures containment this state is "within"
        canister_within = sutils.within(
            sim, canister_object, keypoint_vote_threshold=6
        )
        assert len(canister_within) == 1
        assert basket_object.object_id in canister_within
        # when center is part of the vote with threshold 6, this state is not "within"
        canister_within = sutils.within(
            sim,
            canister_object,
            keypoint_vote_threshold=6,
            center_ensures_containment=False,
        )
        assert len(canister_within) == 0

        # when the object is fully contained, it passes the strictest test
        canister_object.translation = mn.Vector3(-2.01639, 1.1, 0.0410867)
        canister_within = sutils.within(
            sim,
            canister_object,
            keypoint_vote_threshold=6,
            center_ensures_containment=False,
        )
        assert len(canister_within) == 1
        assert basket_object.object_id in canister_within


@pytest.mark.skipif(
    not osp.exists("data/hab3_bench_assets/"),
    reason="Requires HSSD benchmark mini dataset.",
)
def test_region_containment_utils():
    sim_settings = default_sim_settings.copy()
    sim_settings[
        "scene_dataset_config_file"
    ] = "data/hab3_bench_assets/hab3-hssd/hab3-hssd.scene_dataset_config.json"
    sim_settings["scene"] = "103997919_171031233"
    hab_cfg = make_cfg(sim_settings)

    with Simulator(hab_cfg) as sim:
        assert len(sim.semantic_scene.regions) > 0

        desk_object = sutils.get_obj_from_handle(
            sim, "41d16010bfc200eb4d71aea6edaf6ad4bc548105_:0000"
        )
        desk_object.motion_type = MotionType.DYNAMIC

        living_room_region_index = 0
        bedroom_region_index = 3

        living_room_region = sim.semantic_scene.regions[
            living_room_region_index
        ]
        bedroom_region = sim.semantic_scene.regions[bedroom_region_index]

        # the desk starts in completely in the living room
        in_livingroom, ratio = sutils.object_in_region(
            sim, desk_object, living_room_region
        )

        assert in_livingroom
        assert (
            ratio > 0
        )  # won't be 1.0 because some AABB corners are inside the floor or walls

        # move the desk most of the way into the bedroom
        desk_object.translation = mn.Vector3(-3.77824, 0.405816, -2.30807)

        # first validate standard region containment
        in_livingroom, livingroom_ratio = sutils.object_in_region(
            sim, desk_object, living_room_region
        )
        in_bedroom, bedroom_ratio = sutils.object_in_region(
            sim, desk_object, bedroom_region
        )

        assert in_livingroom
        assert in_bedroom
        assert (
            abs(livingroom_ratio + bedroom_ratio - 1.0) < 1e-5
        )  # eps for float error
        assert livingroom_ratio > 0
        assert bedroom_ratio > 0
        assert bedroom_ratio > livingroom_ratio

        # compute aggregate containment in all scene regions
        all_regions_containment = sutils.get_object_regions(sim, desk_object)

        # this list should be sorted, so bedroom is first
        assert all_regions_containment[0][0] == bedroom_region_index
        assert (
            abs(all_regions_containment[0][1] - bedroom_ratio) < 1e-5
        )  # eps for float error
        assert all_regions_containment[1][0] == living_room_region_index
        assert (
            abs(all_regions_containment[1][1] - livingroom_ratio) < 1e-5
        )  # eps for float error
        assert len(all_regions_containment) == 2

        # "center_only" excludes the livingroom
        in_livingroom, livingroom_ratio = sutils.object_in_region(
            sim, desk_object, living_room_region, center_only=True
        )
        in_bedroom, bedroom_ratio = sutils.object_in_region(
            sim, desk_object, bedroom_region, center_only=True
        )

        assert not in_livingroom
        assert in_bedroom
        assert livingroom_ratio == 0
        assert bedroom_ratio == 1.0

        # "containment_threshold" greater than half excludes the livingroom
        in_livingroom, livingroom_ratio = sutils.object_in_region(
            sim, desk_object, living_room_region, containment_threshold=0.51
        )
        in_bedroom, bedroom_ratio = sutils.object_in_region(
            sim, desk_object, bedroom_region, containment_threshold=0.51
        )

        assert not in_livingroom
        assert in_bedroom
        assert livingroom_ratio + bedroom_ratio == 1.0
        assert livingroom_ratio > 0
        assert livingroom_ratio < 0.51
        assert bedroom_ratio > 0.51


@pytest.mark.skipif(
    not built_with_bullet,
    reason="Raycasting API requires Bullet physics.",
)
@pytest.mark.skipif(
    not osp.exists("data/replica_cad/"),
    reason="Requires ReplicaCAD dataset.",
)
def test_ao_open_close_queries():
    sim_settings = default_sim_settings.copy()
    sim_settings[
        "scene_dataset_config_file"
    ] = "data/replica_cad/replicaCAD.scene_dataset_config.json"
    sim_settings["scene"] = "apt_0"
    hab_cfg = make_cfg(sim_settings)
    with Simulator(hab_cfg) as sim:
        # for revolute hinge doors
        fridge = sutils.get_obj_from_handle(sim, "fridge_:0000")

        # for prismatic drawers
        kitchen_counter = sutils.get_obj_from_handle(
            sim, "kitchen_counter_:0000"
        )

        # for prismatic (sliding) doors
        cabinet = sutils.get_obj_from_handle(sim, "cabinet_:0000")

        objects = [fridge, kitchen_counter, cabinet]
        for obj in objects:
            for link_id in obj.get_link_ids():
                if obj.get_link_joint_type(link_id) in [
                    JointType.Revolute,
                    JointType.Prismatic,
                ]:
                    if (
                        "cabinet" in obj.handle
                        and "right_door" in obj.get_link_name(link_id)
                    ):
                        print(
                            "TODO: Skipping 'cabinet's right_door' link because it does not follow conventions. Axis should be inverted."
                        )
                        continue
                    assert sutils.link_is_closed(
                        obj, link_id
                    ), f"Object '{obj.handle}' link {link_id}:'{obj.get_link_name(link_id)}' should be closed with state {sutils.get_link_normalized_joint_position(obj, link_id)}."
                    assert not sutils.link_is_open(obj, link_id)
                    sutils.open_link(obj, link_id)
                    assert sutils.link_is_open(obj, link_id)
                    assert not sutils.link_is_closed(obj, link_id)
                    sutils.close_link(obj, link_id)
                    assert not sutils.link_is_open(obj, link_id)
                    assert sutils.link_is_closed(obj, link_id)
                    # test an intermediate position and the normalized setter utils
                    sutils.set_link_normalized_joint_position(
                        obj, link_id, 0.35
                    )
                    assert not sutils.link_is_open(obj, link_id)
                    assert not sutils.link_is_closed(obj, link_id)
                    assert sutils.link_is_open(obj, link_id, threshold=0.34)
                    assert sutils.link_is_closed(obj, link_id, threshold=0.36)
                    assert (
                        abs(
                            sutils.get_link_normalized_joint_position(
                                obj, link_id
                            )
                            - 0.35
                        )
                        < 1e-5
                    )
                    sutils.close_link(obj, link_id)  # debug reset state

        ################################
        # test default link functionality

        # test computing the default link
        default_link = sutils.get_ao_default_link(fridge)
        assert default_link is None
        default_link = sutils.get_ao_default_link(
            fridge, compute_if_not_found=True
        )
        assert default_link == 1
        assert fridge.user_attributes.get("default_link") == 1
        default_link = sutils.get_ao_default_link(
            kitchen_counter, compute_if_not_found=True
        )
        assert default_link == 6

        # NOTE: sim bug here doesn't break the feature
        # test setting the default link in template metadata
        fridge_template = fridge.creation_attributes
        assert fridge_template.get_user_config().get("default_link") is None
        fridge_template.get_user_config().set("default_link", 0)
        assert fridge_template.get_user_config().get("default_link") == 0
        sim.metadata_mediator.ao_template_manager.register_template(
            fridge_template, "new_fridge_template"
        )
        new_fridge_template_check = (
            sim.metadata_mediator.ao_template_manager.get_template_by_handle(
                "new_fridge_template"
            )
        )
        assert (
            new_fridge_template_check.get_user_config().get("default_link")
            == 0
        )
        new_fridge = sim.get_articulated_object_manager().add_articulated_object_by_template_handle(
            "new_fridge_template"
        )
        assert new_fridge is not None
        default_link = sutils.get_ao_default_link(
            fridge, compute_if_not_found=True
        )
        assert default_link == 1
        new_default_link = sutils.get_ao_default_link(
            new_fridge, compute_if_not_found=True
        )

        # "default_link" should get copied over after instantiation if set in the template programmatically.
        assert new_default_link == 0

        # test setting the default link in instance metadata
        fridge.user_attributes.set("default_link", 0)
        assert fridge.user_attributes.get("default_link") == 0
        default_link = sutils.get_ao_default_link(
            fridge, compute_if_not_found=True
        )
        assert fridge.user_attributes.get("default_link") == 0


@pytest.mark.skipif(
    not built_with_bullet,
    reason="Collision detection API requires Bullet physics.",
)
@pytest.mark.skipif(
    not osp.exists("data/replica_cad/"),
    reason="Requires ReplicaCAD dataset.",
)
def test_ontop_util():
    sim_settings = default_sim_settings.copy()
    sim_settings[
        "scene_dataset_config_file"
    ] = "data/replica_cad/replicaCAD.scene_dataset_config.json"
    sim_settings["scene"] = "apt_0"
    hab_cfg = make_cfg(sim_settings)
    with Simulator(hab_cfg) as sim:
        # a rigid object to test
        table_object = sutils.get_obj_from_handle(
            sim, "frl_apartment_table_02_:0000"
        )

        # an articulated object to test
        counter_object = sutils.get_obj_from_handle(
            sim, "kitchen_counter_:0000"
        )

        # the link to test
        drawer_link_id = 7

        drawer_link_object_ids = [
            obj_id
            for obj_id in counter_object.link_object_ids.keys()
            if counter_object.link_object_ids[obj_id] == drawer_link_id
        ]
        assert len(drawer_link_object_ids) == 1
        drawer_link_object_id = drawer_link_object_ids[0]

        # open the drawer
        drawer_link_dof = counter_object.get_link_joint_pos_offset(
            drawer_link_id
        )
        joint_positions = counter_object.joint_positions
        joint_positions[drawer_link_dof] = 0.5
        counter_object.joint_positions = joint_positions

        # drop an object into the open drawer
        container_object = sutils.get_obj_from_handle(
            sim, "frl_apartment_kitchen_utensil_08_:0000"
        )
        container_object.translation = mn.Vector3(-1.7, 0.6, 0.2)

        # in the initial state:
        # objects are on the table
        assert sutils.ontop(sim, table_object, True) == [
            102,
            103,
            51,
            52,
            53,
            55,
        ]
        assert sutils.ontop(sim, table_object, False) == sutils.ontop(
            sim, table_object.object_id, False
        )
        # objects about the counter are floating slightly and don't register
        assert len(sutils.ontop(sim, counter_object, False)) == 0
        assert len(sutils.ontop(sim, drawer_link_object_id, False)) == 0

        # after some simulation, object settle onto the counter and drawer
        sim.step_physics(0.75)

        # objects are on the table
        # NOTE: we only do collision detection on the first query after a state change
        assert sutils.ontop(sim, table_object, True) == [
            102,
            103,
            51,
            52,
            53,
            55,
        ]
        assert sutils.ontop(sim, table_object, False) == sutils.ontop(
            sim, table_object.object_id, False
        )
        on_counter = sutils.ontop(sim, counter_object, False)
        assert on_counter == [
            65,
            1,
            67,
            68,
            69,
            70,
            71,
            72,
            66,
            81,
            82,
            83,
            84,
            85,
            86,
            87,
            63,
        ]
        assert container_object.object_id in on_counter
        assert sutils.ontop(sim, drawer_link_object_id, False) == [
            container_object.object_id
        ]
        assert sutils.ontop(sim, counter_object.object_id, False) == on_counter


@pytest.mark.skipif(
    not built_with_bullet,
    reason="Collision detection API requires Bullet physics.",
)
@pytest.mark.skipif(
    not osp.exists("data/replica_cad/"),
    reason="Requires ReplicaCAD dataset.",
)
def test_on_floor_and_next_to():
    sim_settings = default_sim_settings.copy()
    sim_settings[
        "scene_dataset_config_file"
    ] = "data/replica_cad/replicaCAD.scene_dataset_config.json"
    sim_settings["scene"] = "apt_0"
    hab_cfg = make_cfg(sim_settings)
    with Simulator(hab_cfg) as sim:
        all_objects = sutils.get_all_object_ids(sim)
        ao_link_map = sutils.get_ao_link_id_map(sim)

        for obj_id, handle in all_objects.items():
            obj = sutils.get_obj_from_id(sim, obj_id, ao_link_map)
            if not obj.is_articulated:
                obj_on_floor = sutils.on_floor(
                    sim, obj, ao_link_map=ao_link_map
                )
                print(f"{handle}: {obj_on_floor}")
                # check a known set of relationships in this scene
                if "plant" in handle:
                    assert obj_on_floor, "All potted plants are on the floor."
                if "lamp" in handle:
                    assert (
                        not obj_on_floor
                    ), "All lamps are on furniture off the floor."
                if "chair" in handle:
                    assert (
                        obj_on_floor
                    ), "All chairs are furniture on the floor."
                if "rug" in handle:
                    assert obj_on_floor, "All rugs are on the floor."
                if "remote-control" in handle:
                    assert (
                        not obj_on_floor
                    ), "All remotes are on furniture, off the floor."
                if "picture" in handle:
                    assert (
                        not obj_on_floor
                    ), "All pictures are on furniture, off the floor."
                if "beanbag" in handle:
                    assert (
                        obj_on_floor
                    ), "All beanbags are furniture on the floor."

        # also test the regularized distance functions directly
        table_object = sutils.get_obj_from_handle(
            sim, "frl_apartment_table_02_:0000"
        )
        objects_in_table = [
            "frl_apartment_choppingboard_02_:0000",
            "frl_apartment_kitchen_utensil_01_:0000",
            "frl_apartment_pan_01_:0000",
            "frl_apartment_bowl_07_:0000",
            "frl_apartment_kitchen_utensil_05_:0000",
        ]
        for obj_handle in objects_in_table:
            obj = sutils.get_obj_from_handle(sim, obj_handle)
            l2_dist = (obj.translation - table_object.translation).length()
            reg_dist = sutils.size_regularized_object_distance(
                sim,
                table_object.object_id,
                obj.object_id,
                ao_link_map,
            )
            # since the objects are in the shelves of the table, regularized distance is o
            assert reg_dist == 0, f"{obj_handle}"
            # L2 distance is computed from CoM or anchor point and will be non-zero
            assert l2_dist > 0

        objects_on_table = [
            "frl_apartment_lamp_02_:0001",
            "frl_apartment_lamp_02_:0000",
        ]
        for obj_handle in objects_on_table:
            obj = sutils.get_obj_from_handle(sim, obj_handle)
            l2_dist = (obj.translation - table_object.translation).length()
            reg_dist = sutils.size_regularized_object_distance(
                sim,
                table_object.object_id,
                obj.object_id,
                ao_link_map,
            )
            # since the objects are "on" the table surface, regularized distance is small, but non-zero
            assert reg_dist != 0, f"{obj_handle}"
            assert reg_dist < 0.1, f"{obj_handle}"
            # L2 distance is computed from CoM or anchor point and will be non-zero
            assert l2_dist > 0
            assert l2_dist > reg_dist

        # test distance between two large neighboring objects
        sofa = sutils.get_obj_from_handle(sim, "frl_apartment_sofa_:0000")
        shelf = sutils.get_obj_from_handle(
            sim, "frl_apartment_wall_cabinet_01_:0000"
        )
        reg_dist = sutils.size_regularized_object_distance(
            sim, sofa.object_id, shelf.object_id, ao_link_map
        )
        assert (
            reg_dist < 0.1
        ), "sofa and shelf should be very close heuristically"
        l2_dist = (sofa.translation - shelf.translation).length()
        assert l2_dist > reg_dist
        assert (
            l2_dist > 1.0
        ), "sofa is more than 1 meter from center to end, so l2 distance to neighbors is typically large."

        # test the bb size heuristic with known directions
        sofa.transformation = mn.Matrix4.identity_init()
        sofa_bb, transform = sutils.get_bb_for_object_id(
            sim, sofa.object_id, ao_link_map
        )
        assert transform == mn.Matrix4.identity_init()
        # check the obvious axis-aligned vectors
        for axis in range(3):
            vec = mn.Vector3()
            vec[axis] = 1.0
            axis_size_along, _center = sutils.get_obj_size_along(
                sim, sofa.object_id, vec, ao_link_map
            )
            assert axis_size_along == sofa_bb.size()[axis] / 2.0

        # test next_to logics

        # NOTE: using ids because they can represent links also, providing handles for readability
        next_to_object_pairs = [
            (3, 4),  # neighboring trashcans
            (102, 103),  # lamps on the table
            (145, 50),  # table and cabinet furniture
            (40, 38),  # books on the same shelf
            (22, 23),  # two neighboring lounge chairs
            (11, 13),  # two neighboring Sofa pillows
            (51, 52),  # two neighboring objects on the table
            (141, 142),  # two neighboring drawers in the chest of drawers
            (131, 132),  # two neighboring cabinet doors
            (77, 78),  # two neighboring spice jars
            (77, 79),  # two skip neighboring spice jars
            (77, 80),  # two double-skip neighboring spice jars
        ]
        not_next_to_object_pairs = [
            (36, 38),  # books on different shelves
            (141, 140),  # two non-neighboring drawers in the chest of drawers
            (11, 14),  # sofa pillows on opposite sides
            (51, 53),  # two objects on different table shelves
            (129, 132),  # two non-neighboring cabinet doors
            (17, 20),  # potted plant and coffee table
        ]
        for ix, (obj_a_id, obj_b_id) in enumerate(next_to_object_pairs):
            assert sutils.obj_next_to(
                sim,
                obj_a_id,
                obj_b_id,
                ao_link_map=ao_link_map,
                vertical_padding=0,
            ), f"Objects with ids {obj_a_id} and {obj_b_id} at test pair index {ix} should be 'next to' one another."
        for ix, (obj_a_id, obj_b_id) in enumerate(not_next_to_object_pairs):
            assert not sutils.obj_next_to(
                sim,
                obj_a_id,
                obj_b_id,
                ao_link_map=ao_link_map,
                vertical_padding=0,
            ), f"Objects with ids {obj_a_id} and {obj_b_id} at test pair index {ix} should not be 'next to' one another."

        # NOTE: the drawers are next_to one another with default 10cm vertical padding
        assert sutils.obj_next_to(
            sim,
            140,
            141,
            ao_link_map=ao_link_map,
        ), "The drawers should be 'next to' one another with vertical padding."
