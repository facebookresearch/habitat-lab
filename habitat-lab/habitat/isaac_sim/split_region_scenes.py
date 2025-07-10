#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Union

from tqdm import tqdm

import habitat.sims.habitat_simulator.sim_utilities as sutils
from habitat.datasets.rearrange.samplers.receptacle import (
    find_receptacles,
    get_recs_from_filter_file,
    get_scene_rec_filter_filepath,
)
from habitat.isaac_sim.isaac_prim_utils import magnum_quat_to_list_wxyz
from habitat_sim import Simulator
from habitat_sim.metadata import MetadataMediator
from habitat_sim.physics import (
    ManagedArticulatedObject,
    ManagedRigidObject,
    MotionType,
)
from habitat_sim.utils.settings import default_sim_settings, make_cfg


def load_scene_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def get_obj_in_regions(
    sim: Simulator,
) -> Dict[int, List[Union[ManagedRigidObject, ManagedArticulatedObject]]]:
    """
    classifies all objects in the scene by their region and returns them in a dictionary keyed by region index.
    """
    regions_to_obj = defaultdict(list)
    for obj in sutils.get_all_objects(sim):
        obj_regions = sutils.get_object_regions(sim, obj)
        for regix, _ in obj_regions:
            regions_to_obj[regix].append(obj)
    return regions_to_obj


def write_region_scene(
    region_objects: List[Union[ManagedRigidObject, ManagedArticulatedObject]],
    base_scene_json: Dict[str, Any],
    outfile: str,
):
    """
    Writes out a new scene JSON file for each region with the objects classified by region.
    """
    scene_copy = base_scene_json.copy()
    # clear the object instances
    scene_copy["object_instances"] = []
    scene_copy["articulated_object_instances"] = []
    # TODO: write new object instances for each region
    mt_strings = {
        MotionType.STATIC: "static",
        MotionType.KINEMATIC: "kinematic",
        MotionType.DYNAMIC: "dynamic",
    }
    for obj in region_objects:
        if isinstance(obj, ManagedRigidObject):
            local_translation = obj.transformation.transform_point(
                obj.com_correction
            )
            obj_instance = {
                "template_name": obj.handle.split("_:")[0],
                "motion_type": mt_strings[obj.motion_type],
                "non_uniform_scale": list(obj.scale),
                # "translation": list(obj.translation),
                "translation": list(local_translation),
                "rotation": magnum_quat_to_list_wxyz(obj.rotation),
                # "translation_origin": "COM"
            }
            scene_copy["object_instances"].append(obj_instance)
        elif isinstance(obj, ManagedArticulatedObject):
            obj_instance = {
                "template_name": obj.handle.split("_:")[0],
                "motion_type": mt_strings[obj.motion_type],
                "base_type": "fixed",
                "translation": list(obj.translation),
                "rotation": magnum_quat_to_list_wxyz(obj.rotation),
                "uniform_scale": obj.global_scale,
                "translation_origin": "COM",
            }
            scene_copy["articulated_object_instances"].append(obj_instance)
    with open(outfile, "w") as f:
        json.dump(scene_copy, f, indent=2)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", dest="dataset", type=str, default=None)
    parser.add_argument("--outdir", type=str, default="region_scenes_out/")
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
        source_scene_json = load_scene_json(scene_path)
        scene_short_name = scene_path.split("/")[-1].split(
            ".scene_instance.json"
        )[0]
        cfg.sim_cfg.scene_id = scene_path
        with Simulator(cfg) as sim:
            # classify objects by region
            reg_to_obj = get_obj_in_regions(sim)

            # sort out objects with valid receptacles so at least one exists in each written scene
            rec_filter_filepath = get_scene_rec_filter_filepath(
                sim.metadata_mediator, sim.curr_scene_name
            )
            active_rec_unames = get_recs_from_filter_file(
                rec_filter_filepath, filter_types=["active"]
            )
            receptacle_instances = find_receptacles(sim)
            receptacle_instances = [
                r
                for r in receptacle_instances
                if r.unique_name in active_rec_unames
            ]
            active_parent_object_handles = set(
                r.parent_object_handle for r in receptacle_instances
            )

            # for each region with valid furniture objects, write out a new scenes file
            for regix, objects in reg_to_obj.items():
                if len(objects) == 0:
                    continue
                some_in_active = False
                for obj in objects:
                    if obj.handle in active_parent_object_handles:
                        some_in_active = True
                        break
                if not some_in_active:
                    print(
                        f"Skipping region {regix} in scene {scene_short_name} as no objects are in active receptacles."
                    )
                    continue
                outfile = os.path.join(
                    args.outdir,
                    f"{scene_short_name}_{regix}.scene_instance.json",
                )
                write_region_scene(objects, source_scene_json, outfile)
                print(f"Wrote region {regix} to {outfile}")


if __name__ == "__main__":
    main()
