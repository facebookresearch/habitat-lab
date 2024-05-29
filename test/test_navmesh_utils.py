#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp

import magnum as mn
import pytest
from omegaconf import DictConfig

import habitat.datasets.rearrange.navmesh_utils as nav_utils
import habitat.sims.habitat_simulator.sim_utilities as sutils
from habitat.articulated_agents.robots.spot_robot import SpotRobot
from habitat.tasks.rearrange.utils import general_sim_collision
from habitat_sim import Simulator, built_with_bullet
from habitat_sim.utils.settings import default_sim_settings, make_cfg


@pytest.mark.skipif(
    not built_with_bullet,
    reason="Raycasting API requires Bullet physics.",
)
@pytest.mark.skipif(
    not osp.exists("data/replica_cad/"),
    reason="Requires ReplicaCAD dataset.",
)
@pytest.mark.skipif(
    not osp.exists("data/robots/hab_spot_arm/"),
    reason="Requires Spot robot embodiment.",
)
@pytest.mark.parametrize("scene_id", ["apt_0", "v3_sc0_staging_00"])
def test_unoccluded_snapping_utils(scene_id):
    sim_settings = default_sim_settings.copy()
    sim_settings[
        "scene_dataset_config_file"
    ] = "data/replica_cad/replicaCAD.scene_dataset_config.json"
    sim_settings["scene"] = scene_id
    hab_cfg = make_cfg(sim_settings)
    with Simulator(hab_cfg) as sim:
        # explicitly load the navmesh
        # NOTE: in apt_0, navmesh does not include furniture, robot should find valid placements anyway with collision checking
        sim.pathfinder.load_nav_mesh(
            f"data/replica_cad/navmeshes/{scene_id}.navmesh"
        )

        # setup for visual debugging
        # sim.navmesh_visualization = True
        # from habitat.sims.habitat_simulator.debug_visualizer import (
        #    DebugVisualizer,
        # )
        # dbv = DebugVisualizer(sim)

        # add the robot to the world via the wrapper
        robot_path = "data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf"
        agent_config = DictConfig({"articulated_agent_urdf": robot_path})
        spot = SpotRobot(agent_config, sim)
        spot.reconfigure()
        spot.update()

        # get the table in the middle of the room
        table_object = sutils.get_obj_from_handle(
            sim, "frl_apartment_table_02_:0000"
        )

        agent_object_ids = [spot.sim_obj.object_id] + [
            *spot.sim_obj.link_object_ids.keys()
        ]

        for _ in range(100):
            sampled_orientations = []
            for orientation_noise_level in [0, 0.1, 0.2, 0.5]:
                # do an embodied snap
                (
                    snap_point,
                    orientation,
                    success,
                ) = nav_utils.embodied_unoccluded_navmesh_snap(
                    target_position=table_object.translation,
                    height=1.3,
                    sim=sim,
                    target_object_ids=[table_object.object_id],
                    ignore_object_ids=agent_object_ids,
                    agent_embodiment=spot,
                    orientation_noise=orientation_noise_level,
                )

                # dbv.peek(spot.sim_obj, peek_all_axis=True).show()
                # breakpoint()

                # should always succeed here
                assert success
                assert orientation not in sampled_orientations
                sampled_orientations.append(orientation)

                # place the robot at the sampled position
                spot.base_pos = snap_point
                spot.base_rot = orientation

                # check that the robot is not in collision
                sim.perform_discrete_collision_detection()
                _, details = general_sim_collision(sim, spot)
                assert details.robot_scene_colls == 0

                table_occluded = nav_utils.snap_point_is_occluded(
                    target=table_object.translation,
                    snap_point=snap_point,
                    height=1.3,
                    sim=sim,
                    target_object_ids=[table_object.object_id],
                    ignore_object_ids=agent_object_ids,
                )

                assert not table_occluded

        # try some expected failures

        # 1. a point far off the navmesh
        (
            snap_point,
            orientation,
            success,
        ) = nav_utils.embodied_unoccluded_navmesh_snap(
            target_position=mn.Vector3(1000, 0, 0),
            height=1.3,
            sim=sim,
            target_object_ids=[table_object.object_id],
            ignore_object_ids=agent_object_ids,
            agent_embodiment=spot,
            orientation_noise=orientation_noise_level,
        )

        assert snap_point is None
        assert orientation is None
        assert not success

        #########################################
        # 2. pass in nan vectors, expect ValueError
        caught_error = False
        try:
            (
                snap_point,
                orientation,
                success,
            ) = nav_utils.embodied_unoccluded_navmesh_snap(
                target_position=mn.Vector3(
                    float("nan"), float("nan"), float("nan")
                ),
                height=1.3,
                sim=sim,
                target_object_ids=[table_object.object_id],
                ignore_object_ids=agent_object_ids,
                agent_embodiment=spot,
                orientation_noise=orientation_noise_level,
            )
        except ValueError:
            caught_error = True
        assert caught_error

        caught_error = False
        try:
            _ = nav_utils.snap_point_is_occluded(
                target=mn.Vector3(float("nan"), float("nan"), float("nan")),
                snap_point=mn.Vector3(1, 0, 0),
                height=1.3,
                sim=sim,
                target_object_ids=[table_object.object_id],
                ignore_object_ids=agent_object_ids,
            )
        except ValueError:
            caught_error = True
        assert caught_error

        caught_error = False
        try:
            _ = nav_utils.snap_point_is_occluded(
                target=mn.Vector3(1, 0, 0),
                snap_point=mn.Vector3(
                    float("nan"), float("nan"), float("nan")
                ),
                height=1.3,
                sim=sim,
                target_object_ids=[table_object.object_id],
                ignore_object_ids=agent_object_ids,
            )
        except ValueError:
            caught_error = True
        assert caught_error
