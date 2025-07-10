#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from tqdm import tqdm

from habitat_sim import Simulator
from habitat_sim.metadata import MetadataMediator
from habitat_sim.nav import NavMeshSettings
from habitat_sim.physics import MotionType
from habitat_sim.utils.settings import default_sim_settings, make_cfg


def navmesh_config_and_recompute(sim: Simulator, outfile: str) -> None:
    """
    This method computes and saves a new navmesh for the current scene based with STATIC aos.
    """

    navmesh_settings = NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.agent_height = 1.3
    navmesh_settings.agent_radius = 0.3
    navmesh_settings.include_static_objects = True
    navmesh_settings.cell_height = 0.01

    cached_motion_types = {}
    aom = sim.get_articulated_object_manager()
    for ao in aom.get_objects_by_handle_substring().values():
        cached_motion_types[ao.handle] = ao.motion_type
        ao.motion_type = MotionType.STATIC

    sim.recompute_navmesh(
        sim.pathfinder,
        navmesh_settings,
    )

    sim.pathfinder.save_nav_mesh(outfile)
    print(f"Saved navmesh to {outfile}")

    for ao in aom.get_objects_by_handle_substring().values():
        ao.motion_type = cached_motion_types[ao.handle]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", dest="dataset", type=str, default=None)
    parser.add_argument(
        "--outdir", type=str, default="data/hssd-hab/navmeshes/"
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        type=str,
        help="A subset of scene names to process. Limits the iteration to less than the full set of scenes.",
        default=None,
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    sim_settings = default_sim_settings.copy()
    sim_settings["scene_dataset_config_file"] = args.dataset
    sim_settings["scene"] = "NONE"
    sim_settings["enable_hbao"] = True
    cfg = make_cfg(sim_settings)

    # pre-initialize a MetadataMediator to iterate over scenes
    mm = MetadataMediator()
    mm.active_dataset = args.dataset
    cfg.metadata_mediator = mm

    target_scenes = mm.get_scene_handles()
    if args.scenes is not None:
        target_scenes = args.scenes

    for scene_path in tqdm(target_scenes):
        cfg.sim_cfg.scene_id = scene_path
        with Simulator(cfg) as sim:
            scene_name = sim.curr_scene_name.split("/")[-1].split(
                ".scene_instance.json"
            )[0]
            navmesh_filepath = os.path.join(
                args.outdir, f"{scene_name}.navmesh"
            )
            navmesh_config_and_recompute(sim, navmesh_filepath)


if __name__ == "__main__":
    main()
