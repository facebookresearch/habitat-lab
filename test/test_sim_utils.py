#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp

import magnum as mn
import pytest

from habitat.sims.habitat_simulator.sim_utilities import (
    bb_ray_prescreen,
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
    sim_settings["sensor_height"] = 0
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
