#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import time
from multiprocessing import Manager, Pool
from pathlib import Path
from typing import Callable, List, Set

import decimate

OUTPUT_DIR = "data/hitl_simplified/data/"
OMIT_BLACK_LIST = False
OMIT_GRAY_LIST = False
PROCESS_COUNT = os.cpu_count()


class Job:
    source_path: str
    dest_path: str
    simplify: bool


class Config:
    # Increase logging verbosity.
    verbose: bool = False
    # Activate multiprocessing. Disable when debugging.
    use_multiprocessing: bool = False


def file_is_scene_config(filepath: str) -> bool:
    """
    Return whether or not the file is an scene_instance.json
    """
    return filepath.endswith(".scene_instance.json")


def file_is_glb(filepath: str) -> bool:
    """
    Return whether or not the file is a glb.
    """
    return filepath.endswith(".glb")


def find_files(
    root_dir: str, discriminator: Callable[[str], bool]
) -> List[str]:
    """
    Recursively find all filepaths under a root directory satisfying a particular constraint as defined by a discriminator function.

    :param root_dir: The root directory for the recursive search.
    :param discriminator: The discriminator function which takes a filepath and returns a bool.

    :return: The list of all absolute filepaths found satisfying the discriminator.
    """
    filepaths: List[str] = []

    if not os.path.exists(root_dir):
        print(" Directory does not exist: " + str(dir))
        return filepaths

    for entry in os.listdir(root_dir):
        entry_path = os.path.join(root_dir, entry)
        if os.path.isdir(entry_path):
            sub_dir_filepaths = find_files(entry_path, discriminator)
            filepaths.extend(sub_dir_filepaths)
        # apply a user-provided discriminator function to cull filepaths
        elif discriminator(entry_path):
            filepaths.append(entry_path)
    return filepaths


def get_model_ids_from_scene_instance_json(filepath: str) -> List[str]:
    """
    Scrape a list of all unique model ids from the scene instance file.
    """
    assert filepath.endswith(
        ".scene_instance.json"
    ), "Must be a scene instance JSON."

    model_ids = []

    with open(filepath, "r") as f:
        scene_conf = json.load(f)
        if "object_instances" in scene_conf:
            for obj_inst in scene_conf["object_instances"]:
                model_ids.append(obj_inst["template_name"])
        else:
            print(
                "No object instances field detected, are you sure this is scene instance file?"
            )

    print(f" {filepath} has {len(model_ids)} object instances.")
    model_ids = list(set(model_ids))
    print(f" {filepath} has {len(model_ids)} unique objects.")

    return model_ids


def validate_jobs(jobs: List[Job]):
    for job in jobs:
        assert Path(job.source_path).exists
        assert job.dest_path != None and job.dest_path != ""


def process_model(args):
    job, counter, lock, total_models, verbose = args

    if os.path.isfile(job.dest_path):
        print(f"Skipping:   {job.source_path}")
        result = {"status": "skipped"}
        return result

    print(f"Processing: {job.source_path}")

    # Create all necessary subdirectories
    os.makedirs(os.path.dirname(job.dest_path), exist_ok=True)

    try:
        source_tris, target_tris, simplified_tris = decimate.decimate(
            inputFile=job.source_path,
            outputFile=job.dest_path,
            quiet=not verbose,
            verbose=verbose,
            sloppy=False,
            simplify=job.simplify,
        )
    except Exception:
        try:
            print(
                f"Unable to decimate: {job.source_path}. Trying without decimation."
            )
            source_tris, target_tris, simplified_tris = decimate.decimate(
                inputFile=job.source_path,
                outputFile=job.dest_path,
                quiet=not verbose,
                verbose=verbose,
                simplify=False,
            )
        except Exception:
            print(f"Unable to decimate: {job.source_path}")
            result = {"status": "error"}
            return result

    print(
        f"source_tris: {source_tris}, target_tris: {target_tris}, simplified_tris: {simplified_tris}"
    )

    result = {
        "source_tris": source_tris,
        "simplified_tris": simplified_tris,
        "source_path": job.source_path,
        "status": "ok",
    }

    if simplified_tris > target_tris * 2 and simplified_tris > 3000:
        result["list_type"] = "black"
        if OMIT_BLACK_LIST:
            os.remove(job.dest_path)
    elif simplified_tris > 4000:
        result["list_type"] = "gray"
        if OMIT_GRAY_LIST:
            os.remove(job.dest_path)
    else:
        result["list_type"] = None

    with lock:
        counter.value += 1
        print(
            f"{counter.value} out of {total_models} models have been processed so far"
        )

    return result


def simplify_models(jobs: List[Job], config: Config):
    start_time = time.time()
    total_source_tris = 0
    total_simplified_tris = 0
    black_list = []
    gray_list = []
    black_list_tris = 0
    gray_list_tris = 0
    total_skipped = 0
    total_error = 0

    validate_jobs(jobs)
    total_models = len(jobs)

    # Initialize counter and lock
    manager = Manager()
    counter = manager.Value("i", 0)
    lock = manager.Lock()

    total_models = len(jobs)

    # Pair up the model paths with the counter and lock
    args_lists = [
        (job, counter, lock, total_models, config.verbose) for job in jobs
    ]

    results = []

    if config.use_multiprocessing:
        max_processes = PROCESS_COUNT
        with Pool(processes=min(max_processes, total_models)) as pool:
            results = list(pool.map(process_model, args_lists))
    else:
        for args in args_lists:
            results.append(process_model(args))

    for result in results:
        if result["status"] == "ok":
            total_source_tris += result["source_tris"]
            total_simplified_tris += result["simplified_tris"]
            if result["list_type"] == "black":
                black_list.append(result["source_path"])
                black_list_tris += result["simplified_tris"]
            elif result["list_type"] == "gray":
                gray_list.append(result["source_path"])
                gray_list_tris += result["simplified_tris"]
        elif result["status"] == "error":
            total_error += 1
        elif result["status"] == "skipped":
            total_skipped += 1

    if total_skipped > 0:
        print(f"Skipped {total_skipped} files.")
    if total_error > 0:
        print(f"Skipped {total_error} files due to processing errors.")
    print(
        f"Reduced total vertex count from {total_source_tris} to {total_simplified_tris}"
    )
    print(f"Without black list: {total_simplified_tris - black_list_tris}")
    print(
        f"Without gray and black list: {total_simplified_tris - black_list_tris - gray_list_tris}"
    )

    for i, curr_list in enumerate([black_list, gray_list]):
        print("")
        print("black list" if i == 0 else "gray list" + " = [")
        for item in curr_list:
            print("    " + item + ",")
        print("]")

    elapsed_time = time.time() - start_time
    print(f"Elapsed time (s): {elapsed_time}")
    if not config.use_multiprocessing:
        print("Add --use-multiprocessing to speed-up processing.")


def find_model_paths_in_scenes(hssd_hab_root_dir, scene_ids) -> List[str]:
    model_filepaths: Set[str] = set()
    config_root_dir = os.path.join(hssd_hab_root_dir, "scenes-uncluttered")
    configs = find_files(config_root_dir, file_is_scene_config)
    obj_root_dir = os.path.join(hssd_hab_root_dir, "objects")
    glb_files = find_files(obj_root_dir, file_is_glb)
    render_glbs = [
        f
        for f in glb_files
        if (".collider" not in f and ".filteredSupportSurface" not in f)
    ]

    for filepath in configs:
        # these should be removed, but screen them for now
        if "orig" in filepath:
            print(f"Skipping alleged 'original' instance file {filepath}")
            continue
        for scene_id in scene_ids:
            # NOTE: add the extension back here to avoid partial matches
            if scene_id + ".scene_instance.json" in filepath:
                print(f"filepath '{filepath}' matches scene_id '{scene_id}'")
                model_ids = get_model_ids_from_scene_instance_json(filepath)
                for model_id in model_ids:
                    for render_glb in render_glbs:
                        if model_id + ".glb" in render_glb:
                            if "part" in render_glb and "part" not in model_id:
                                continue
                            model_filepaths.add(render_glb)

    return list(model_filepaths)


def main():
    parser = argparse.ArgumentParser(
        description="Get all .glb render asset files associated with a given scene."
    )
    parser.add_argument(
        "--hssd-hab-root-dir",
        type=str,
        help="Path to the hssd-hab root directory containing 'hssd-hab-uncluttered.scene_dataset_config.json'.",
    )
    parser.add_argument(
        "--hssd-models-root-dir",
        type=str,
        help="Path to hssd-models root directory.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        type=str,
        help="one or more scene ids",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Increase logging verbosity.",
    )
    parser.add_argument(
        "--use-multiprocessing",
        action="store_true",
        default=False,
        help="Enable multiprocessing.",
    )

    args = parser.parse_args()
    config = Config()
    config.verbose = args.verbose
    config.use_multiprocessing = args.use_multiprocessing

    # Force input paths to have a trailing slash
    if args.hssd_hab_root_dir[-1] != "/":
        args.hssd_hab_root_dir += "/"
    if args.hssd_models_root_dir[-1] != "/":
        args.hssd_models_root_dir += "/"

    jobs: List[Job] = []
    scene_ids = list(dict.fromkeys(args.scenes)) if args.scenes else []

    # Define relative directory for the dataset
    # E.g. data/scene_datasets/hssd-hab -> scene_datasets/hssd-hab
    hssd_hab_rel_dir = str(args.hssd_hab_root_dir)[len("data/") :]

    # Add stages
    for scene_id in scene_ids:
        rel_path = os.path.join("stages", scene_id + ".glb")
        job = Job()
        job.source_path = os.path.join(args.hssd_hab_root_dir, rel_path)
        job.dest_path = os.path.join(OUTPUT_DIR, hssd_hab_rel_dir, rel_path)
        job.simplify = False
        jobs.append(job)

    # Add all models contained in the scenes
    scene_models = find_model_paths_in_scenes(
        args.hssd_hab_root_dir, scene_ids
    )
    for scene_model in scene_models:
        rel_path = scene_model[len(args.hssd_hab_root_dir) :]
        if "decomposed" not in scene_model:
            job = Job()
            source_path = os.path.join(args.hssd_models_root_dir, rel_path)
            parts = source_path.split(
                "/objects/"
            )  # Remove 'objects/' from path
            job.source_path = os.path.join(parts[0], parts[1])
            assert len(parts) == 2
            job.dest_path = os.path.join(
                OUTPUT_DIR, hssd_hab_rel_dir, rel_path
            )
            job.simplify = False
            jobs.append(job)
        else:
            job = Job()
            job.source_path = os.path.join(args.hssd_hab_root_dir, rel_path)
            job.dest_path = os.path.join(
                OUTPUT_DIR, hssd_hab_rel_dir, rel_path
            )
            job.simplify = False
            jobs.append(job)

    # Add ycb objects
    for filename in Path("data/objects/ycb/meshes").rglob("*.glb"):
        rel_path = str(filename)[len("data/") :]
        job = Job()
        job.source_path = os.path.join("data", rel_path)
        job.dest_path = os.path.join(OUTPUT_DIR, rel_path)
        job.simplify = False
        jobs.append(job)

    # Add spot models
    for filename in Path("data/robots/hab_spot_arm/meshesColored").rglob(
        "*.glb"
    ):
        rel_path = str(filename)[len("data/") :]
        job = Job()
        job.source_path = os.path.join("data", rel_path)
        job.dest_path = os.path.join(OUTPUT_DIR, rel_path)
        job.simplify = False
        jobs.append(job)

    simplify_models(jobs, config)


if __name__ == "__main__":
    main()
