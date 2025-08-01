import json
import os
import shutil
import time
from dataclasses import dataclass
from multiprocessing import Manager, Pool
from multiprocessing.managers import ValueProxy
from pathlib import Path
from threading import Lock
from typing import Any

import collada
from habitat_dataset_processing import magnum_decimation
from habitat_dataset_processing.configs import Config, Operation
from habitat_dataset_processing.job import Job
from habitat_dataset_processing.util import is_file_collada

METADATA_FILE_VERSION = 1
OMIT_BLACK_LIST = False
OMIT_GRAY_LIST = False
PROCESS_COUNT = os.cpu_count()
LOCAL_GROUP_NAME = "local"
OUTPUT_METADATA_FILE_NAME = "metadata.json"

PROCESSED_EXTENSIONS: list[str] = [".glb", ".gltf"]
SKIPPED_EXTENSIONS: list[str] = [".urdf"]


def create_metadata_file(groups: dict[str, list[str]], output_dir: str):
    """
    Create output metadata file.
    This file may be consumed by custom external asset pipelines (e.g. Unity) to package assets.

    Contents:
    * version: Version of the file format. Bump when breaking backward compatibility.
    * local_group_name: Name of the local group. See 'LOCAL_GROUP_NAME'.
    * groups: list of groups and their assets.
              Groups are used to determine how to package remote assets.
    """
    # Compose file content.
    content: dict[str, Any] = {}
    content["version"] = METADATA_FILE_VERSION
    content["local_group_name"] = LOCAL_GROUP_NAME
    content["groups"] = groups

    # Remove failed files.
    assets_removed = 0
    for group in content["groups"].values():
        assets_to_remove: list[str] = []
        for asset in group:
            asset_path = os.path.join(output_dir, asset)
            if not os.path.exists(asset_path):
                assets_to_remove.append(asset)
                assets_removed += 1
        for asset in assets_to_remove:
            group.remove(asset)
    if assets_removed > 0:
        print(
            f"create_metadata_file: Removed {assets_removed} unprocessed assets."
        )

    # Save file.
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, OUTPUT_METADATA_FILE_NAME)
    with open(output_path, "w") as f:
        json.dump(content, f, ensure_ascii=False)


@dataclass
class AssetProcessorArgs:
    job: Job
    counter: ValueProxy[int]
    lock: Lock
    total_models: int
    verbose: bool


def get_preferred_operation(
    path: str, desired_operation: Operation
) -> Operation:
    """
    Returns `desired_operation` if the model can be processed with this operation.
    Else, returns the preferred operation.
    """
    extension = Path(path).suffix.lower()

    if extension in PROCESSED_EXTENSIONS:
        return desired_operation
    if extension in SKIPPED_EXTENSIONS:
        return Operation.IGNORE
    return Operation.COPY


def transform_collada(path: str):
    """
    Habitat ignores the collada up direction ("ImportColladaIgnoreUpDirection" == "true").
    The collada file needs to be transformed so that the coordinate system matches Habitat.
    """
    assert os.path.exists(path)
    mesh = collada.Collada(path)
    mesh.assetInfo.upaxis = "Y_UP"
    mesh.write(path)


def process_model(args: AssetProcessorArgs):
    job = args.job
    verbose = args.verbose

    result = {
        "source_path": job.source_path,
        "dest_path": job.dest_path,
        "additional_dest_paths": [],
    }

    if os.path.isfile(job.dest_path):
        if verbose:
            print(f"Skipping:   {job.source_path}")
        result["status"] = "skipped"
        return result

    if verbose:
        print(f"Processing: {job.source_path}.")

    operation = get_preferred_operation(job.source_path, job.operation)
    if operation != Operation.IGNORE:
        # Create all necessary subdirectories
        os.makedirs(os.path.dirname(job.dest_path), exist_ok=True)
    else:
        result["status"] = "ignored"
        return result

    if operation == Operation.COPY:
        shutil.copyfile(job.source_path, job.dest_path, follow_symlinks=True)
        result["status"] = "copied"

        if is_file_collada(job.dest_path):
            transform_collada(job.dest_path)

        return result

    elif operation == Operation.PROCESS:
        try:
            (
                source_tris,
                target_tris,
                simplified_tris,
            ) = magnum_decimation.decimate(
                inputFile=job.source_path,
                outputFile=job.dest_path,
                quiet=not verbose,
                verbose=verbose,
                sloppy=False,
                simplify=job.simplify,
            )
        except Exception:
            try:
                magnum_decimation.close()
                print(
                    f"Unable to decimate: {job.source_path}. Trying without decimation."
                )
                (
                    source_tris,
                    target_tris,
                    simplified_tris,
                ) = magnum_decimation.decimate(
                    inputFile=job.source_path,
                    outputFile=job.dest_path,
                    quiet=not verbose,
                    verbose=verbose,
                    simplify=False,
                )
            except Exception:
                magnum_decimation.close()
                print(
                    f"Unable to decimate: {job.source_path}. Copying as-is to output."
                )
                shutil.copyfile(
                    job.source_path, job.dest_path, follow_symlinks=True
                )
                result["status"] = "error"
                return result

    if operation == Operation.PROCESS and job.simplify and verbose:
        print(
            f"source_tris: {source_tris}, target_tris: {target_tris}, simplified_tris: {simplified_tris}"
        )
        result["source_tris"] = source_tris
        result["simplified_tris"] = simplified_tris

    result["status"] = "ok"

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

    with args.lock:
        args.counter.value += 1
        if verbose:
            print(
                f"{args.counter.value} out of {args.total_models} models have been processed so far."
            )

    return result


def process_models(jobs: list[Job], config: Config):
    start_time = time.time()
    total_source_tris = 0
    total_simplified_tris = 0
    black_list = []
    gray_list = []
    black_list_tris = 0
    gray_list_tris = 0
    total_skipped = 0
    total_error = 0
    total_copied = 0

    total_models = len(jobs)

    # Initialize counter and lock
    manager = Manager()
    counter = manager.Value("i", 0)
    lock = manager.Lock()

    total_models = len(jobs)

    # Pair up the model paths with the counter and lock
    args_list: list[AssetProcessorArgs] = []
    for job in jobs:
        args_list.append(
            AssetProcessorArgs(
                job=job,
                counter=counter,
                lock=lock,
                total_models=total_models,
                verbose=config.verbose,
            )
        )

    results = []

    if config.use_multiprocessing:
        max_processes = PROCESS_COUNT
        with Pool(processes=min(max_processes, total_models)) as pool:
            results = list(pool.map(process_model, args_list))
    else:
        for args in args_list:
            results.append(process_model(args))

    for result in results:
        if result["status"] == "ok":
            if "source_tris" in result:
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
        elif result["status"] == "copied":
            total_copied += 1

    if total_skipped > 0:
        print(f"Skipped {total_skipped} files.")
    if total_error > 0:
        print(f"Skipped {total_error} files due to processing errors.")
    if total_copied > 0:
        print(f"Copied {total_copied} files due to unsupported format.")
    if total_source_tris > total_simplified_tris:
        print(
            f"Reduced total vertex count from {total_source_tris} to {total_simplified_tris}."
        )

    elapsed_time = time.time() - start_time
    print(f"Elapsed time (s): {elapsed_time}")
    if not config.use_multiprocessing:
        print("Add --use-multiprocessing to speed-up processing.")
