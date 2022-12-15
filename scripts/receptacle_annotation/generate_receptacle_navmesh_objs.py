#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, List

import git

import habitat_sim
from habitat_sim.utils.settings import default_sim_settings, make_cfg

# get the output directory and data path
repo = git.Repo(".", search_parent_directories=True)
dir_path = repo.working_tree_dir
data_path = os.path.join(dir_path, "data")

# setup the scene settings
sim_settings = default_sim_settings.copy()
sim_settings["enable_physics"] = False  # kinematics only
sim_settings["output_dir"] = "navmeshes/"
sim_settings["navmesh_settings"] = habitat_sim.nav.NavMeshSettings()


def save_navmesh_data(sim: habitat_sim.Simulator, output_dir: str) -> None:
    """
    Iteratively save each navmesh island to a separate OBJ file in the configured output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    if sim.pathfinder.is_loaded:
        for island in range(sim.pathfinder.num_islands):
            vert_data = sim.pathfinder.build_navmesh_vertices(island)
            index_data = sim.pathfinder.build_navmesh_vertex_indices(island)
            export_navmesh_data_to_obj(
                filename=os.path.join(output_dir, f"{island}.obj"),
                vertex_data=vert_data,
                index_data=index_data,
            )
    else:
        print("Cannot save navmesh data, no pathfinder loaded")


def export_navmesh_data_to_obj(
    filename: str, vertex_data: List[Any], index_data: List[int]
) -> None:
    """
    Export triangle mesh data in simple OBJ format.
    NOTE: Could use an exporter framework, but this way is very simple and introduces no dependencies.
    """
    with open(filename, "w") as f:
        file_data = ""
        for vert in vertex_data:
            file_data += (
                "v "
                + str(vert[0])
                + " "
                + str(vert[1])
                + " "
                + str(vert[2])
                + "\n"
            )
        assert len(index_data) % 3 == 0, "must be triangles"
        for ix in range(int(len(index_data) / 3)):
            # NOTE: obj starts indexing at 1
            file_data += (
                "f "
                + str(index_data[ix * 3] + 1)
                + " "
                + str(index_data[ix * 3 + 1] + 1)
                + " "
                + str(index_data[ix * 3 + 2] + 1)
                + "\n"
            )
        f.write(file_data)


def make_cfg_mm(settings):
    """
    Create a Configuration with an attached MetadataMediator for shared dataset access and re-use without instantiating the Simulator object first.
    """
    config = make_cfg(settings)

    # create an attach a MetadataMediator
    mm = habitat_sim.metadata.MetadataMediator(config.sim_cfg)

    return habitat_sim.Configuration(config.sim_cfg, config.agents, mm)


def iteratively_export_all_scenes_navmesh(
    config_with_mm, recompute_navmesh=False
):
    # generate a SceneDataset report for quick investigation
    print("-------------------------------")
    print(config_with_mm.metadata_mediator.dataset_report())
    # list all registered scenes
    print("SCENES")
    for scene_handle in config_with_mm.metadata_mediator.get_scene_handles():
        print(scene_handle)
    # list all registered stages
    print("STAGES")
    stage_handles = (
        config_with_mm.metadata_mediator.stage_template_manager.get_templates_by_handle_substring()
    )
    for stage_handle in stage_handles:
        print(stage_handle)

    failure_log = []
    # iterate over all registered stages to generate navmeshes
    # NOTE: this iteration could be customized to hit a subset of stages or any registered scenes.
    for stage_handle in stage_handles:
        print("=================================================")
        print(f"    {stage_handle}")
        config_with_mm.sim_cfg.scene_id = stage_handle
        if stage_handle == "NONE":
            # skip the empty "NONE" scene which is always present
            continue
        try:
            with habitat_sim.Simulator(config_with_mm) as sim:
                # instance the Simulator with a selected scene/stage and compute/export the navmesh
                stage_filename = stage_handle.split("/")[-1]
                stage_directory = stage_handle[: -len(stage_filename)]
                stage_output_dir = os.path.join(
                    sim_settings["output_dir"],
                    stage_filename.split(".")[0] + "/",
                )
                os.makedirs(stage_output_dir, exist_ok=True)

                # export the render asset path for later use in Blender
                stage_template = sim.metadata_mediator.stage_template_manager.get_template_by_handle(
                    stage_handle
                )
                render_asset_path = os.path.abspath(
                    stage_template.render_asset_handle
                )
                render_asset_record_filepath = os.path.join(
                    stage_output_dir, "render_asset_path.txt"
                )
                with open(render_asset_record_filepath, "w") as f:
                    f.write(render_asset_path)

                # recompute the navmesh if necessary
                if recompute_navmesh or not sim.pathfinder.is_loaded():
                    navmesh_filename = (
                        stage_filename.split(".")[0] + ".navmesh"
                    )
                    sim.recompute_navmesh(
                        sim.pathfinder, sim_settings["navmesh_settings"]
                    )
                    if os.path.exists(stage_directory):
                        sim.pathfinder.save_nav_mesh(
                            stage_output_dir + navmesh_filename
                        )
                    else:
                        failure_log.append(
                            (
                                stage_handle,
                                f"No target directory for navmesh: {stage_directory}",
                            )
                        )
                # export the navmesh OBJs
                save_navmesh_data(sim, output_dir=stage_output_dir)

        except Exception as e:
            failure_log.append((stage_handle, str(e)))
        print("=================================================")
    print(f"Failure log = {failure_log}")
    print(
        f"Tried {len(stage_handles)-1} stages."
    )  # manually decrement the "NONE" scene
    print("-------------------------------")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        dest="dataset",
        type=str,
        help="The SceneDataset config file.",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        default="navmeshes/",
        help="The output directory for the navmesh .OBJ files. Sub-directories will be created for each stage/scene.",
    )
    parser.add_argument(
        "--navmesh-settings",
        dest="navmesh_settings",
        type=str,
        default="",
        help="Optionally provide a path to a navmesh settings JSON file to use instead of the default settings.",
    )
    args, _ = parser.parse_known_args()

    sim_settings["scene_dataset_config_file"] = args.dataset
    sim_settings["output_dir"] = args.output_dir

    # load user defined NavMeshSettings from JSON
    if args.navmesh_settings != "":
        assert os.path.exists(
            args.navmesh_settings
        ), f"Provided NavmeshSettings config file  '{args.navmesh_settings}' not found, aborting."
        assert args.navmesh_settings.endswith(
            ".json"
        ), "args.navmesh_settings must be a NavmeshSettings JSON file."
        sim_settings["navmesh_settings"].read_from_json(args.navmesh_settings)

    iteratively_export_all_scenes_navmesh(
        make_cfg_mm(sim_settings), recompute_navmesh=True
    )
