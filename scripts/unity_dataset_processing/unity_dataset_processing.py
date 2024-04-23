#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from enum import Enum
import gzip
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from multiprocessing import Manager, Pool
from pathlib import Path
from typing import Any, Callable, Dict, List, Set

import decimate

METADATA_FILE_VERSION = 1
OUTPUT_DIR = "data/_processing_output/"
OMIT_BLACK_LIST = False
OMIT_GRAY_LIST = False
PROCESS_COUNT = os.cpu_count()
LOCAL_GROUP_NAME = "local"
OUTPUT_METADATA_FILE_NAME = "metadata.json"


def resolve_relative_path(path: str) -> str:
    """
    Remove './' and '../' from path.
    """
    components = path.split("/")
    output_path: List[str] = []
    for component in components:
        if component == ".":
            continue
        elif component == "..":
            assert (
                len(output_path) > 0
            ), "Relative path escaping out of data folder."
            output_path.pop()
        else:
            output_path.append(component)
    return os.path.join("", *output_path)

class JobType(Enum):
    # Copy the asset as-is, skipping all processing.
    COPY = 1
    # Process the asset to make it compatible with Unity.
    # Enable 'job.decimate' to simplify the model.
    PROCESS = 2


class Job:
    def __init__(
        self,
        asset_path: str,
        output_dir: str,
        groups: List[str],
        simplify: bool,
        job_type: JobType = JobType.PROCESS,
    ):
        self.source_path = resolve_relative_path(asset_path)
        self.dest_path = resolve_relative_path(
            os.path.join(output_dir, asset_path)
        )
        self.groups = groups
        self.simplify = simplify
        self.job_type = job_type

        # If the asset doesn't belong to a group, assign the 'local' group.
        # This group indicates that the asset should be packaged along with the client rather than at a remote location.
        if len(self.groups) == 0:
            groups.append(LOCAL_GROUP_NAME)


class Config:
    # Increase logging verbosity.
    verbose: bool = False
    # Activate multiprocessing. Disable when debugging.
    use_multiprocessing: bool = False


def file_is_object_config(filepath: str) -> bool:
    """
    Return whether or not the file is an object_config.json
    """
    return filepath.endswith(".object_config.json")


def file_is_render_asset_config(filepath: str) -> bool:
    """
    Return if the file is either:
    * object_config.json
    * ao_config.json
    * stage_config.json
    All of these files are guaranteed to have a 'render_asset' field.
    """
    return (
        filepath.endswith(".object_config.json")
        or filepath.endswith(".ao_config.json")
        or filepath.endswith(".stage_config.json")
    )


def file_is_scene_dataset_config(filepath: str) -> bool:
    """
    Return whether or not the file is a scene_dataset_config.json
    """
    return filepath.endswith(".scene_dataset_config.json")


def file_is_scene_config(filepath: str) -> bool:
    """
    Return whether or not the file is a scene_instance.json
    """
    return filepath.endswith(".scene_instance.json")


def file_is_episode_set(filepath: str) -> bool:
    """
    Return whether or not the file is an json.gz
    """
    return filepath.endswith(".json.gz")


def verify_jobs(jobs: List[Job], output_path: str) -> None:
    source_set: Set[str] = set()
    dest_set: Set[str] = set()
    for job in jobs:
        # Check that all assets exist.
        assert Path(
            job.source_path
        ).is_file(), f"Source path is not a file: '{job.source_path}'."
        # Check that all source paths start with "data/".
        prefix = "data/"
        prefix_length = len(prefix)
        assert (
            str(job.source_path)[0:prefix_length] == prefix
        ), f"Source path not in '{prefix}': '{job.source_path}'."
        # Check that all dest paths start with "{output_path}/data/"
        prefix = os.path.join(output_path, "data/")
        prefix_length = len(prefix)
        assert (
            str(job.dest_path)[0:prefix_length] == prefix
        ), f"Dest path not in '{prefix}': '{job.dest_path}'."
        # Check that all job paths are unique.
        # TODO: De-duplicate items from mixed sources.
        """
        assert (
            job.source_path not in source_set
        ), f"Duplicate source asset: '{job.source_path}'."
        assert (
            job.dest_path not in dest_set
        ), f"Duplicate destination asset: '{job.dest_path}'."
        """
        source_set.add(job.source_path)
        dest_set.add(job.dest_path)
        # Check that all paths are resolved (no '.' or '..').
        assert "./" not in job.source_path
        assert "./" not in job.dest_path


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


def search_dataset_render_assets(
    scene_dataset_dir: str, directory_search_patterns: List[str]
) -> Dict[str, List[str]]:
    """
    scene_dataset_dir: "data/scene_datasets/hssd-hab"
    directory_search_patterns: ["objects/*", "objects/decomposed/*"]
    """
    output: Dict[str, List[str]] = {}
    object_config_json_path_cache: set[str] = set()

    for directory_search_pattern in directory_search_patterns:
        # > "data/scene_datasets/hssd-hab/objects/*"
        object_config_search_pattern = os.path.join(
            scene_dataset_dir, directory_search_pattern
        )
        # > "data/scene_datasets/hssd-hab/objects"
        object_config_search_dir = Path(object_config_search_pattern)
        while not Path.is_dir(object_config_search_dir):
            object_config_search_dir = object_config_search_dir.parent
        # Find all object configs in the specified path
        # > "data/scene_datasets/hssd-hab/objects/2/2a0a80bf0b1b247799ed3a49bfdc9f9bf4fcf2b3.object_config.json"
        object_config_json_paths = find_files(
            str(object_config_search_dir), file_is_render_asset_config
        )
        for object_config_json_path in object_config_json_paths:
            # Ignore object if already found.
            if object_config_json_path in object_config_json_path_cache:
                continue
            object_config_json_path_cache.add(object_config_json_path)
            object_config = load_json(object_config_json_path)
            # > "data/scene_datasets/hssd-hab/objects/2"
            object_render_asset_dir = Path(object_config_json_path).parent
            # > "2a0a80bf0b1b247799ed3a49bfdc9f9bf4fcf2b3
            object_template_name = Path(object_config_json_path).stem.split(
                "."
            )[0]
            # Canonical case - the object has a render_asset field.
            if "render_asset" in object_config:
                # > "2a0a80bf0b1b247799ed3a49bfdc9f9bf4fcf2b3.glb"
                object_render_asset_file_name = object_config["render_asset"]
                # > "data/scene_datasets/hssd-hab/objects/2/2a0a80bf0b1b247799ed3a49bfdc9f9bf4fcf2b3.glb"
                object_render_asset_path = os.path.join(
                    object_render_asset_dir, object_render_asset_file_name
                )
                if object_template_name not in output:
                    output[object_template_name] = []
                output[object_template_name].append(object_render_asset_path)
            # Articulated objects may have extra render assets defined in the URDF.
            if "urdf_filepath" in object_config:
                urdf_file_path = os.path.join(
                    object_render_asset_dir, object_config["urdf_filepath"]
                )
                # Hack: Search URDF text directly.
                with open(urdf_file_path, "r") as urdf_file:
                    urdf_text = urdf_file.read()
                    regex = re.compile('filename="(.*?)"')
                    matches = re.findall(regex, urdf_text)
                    for match in matches:
                        render_asset_path = os.path.join(
                            object_render_asset_dir, match
                        )
                        render_asset_path = resolve_relative_path(
                            render_asset_path
                        )
                        if object_template_name not in output:
                            output[object_template_name] = []
                        output[object_template_name].append(render_asset_path)

    return output


class SceneDataset:
    """
    Upon construction, loads a scene_dataset_config.json file and exposes its content.
    """

    # Location of the scene_dataset_config.json.
    # > "data/scene_datasets/hssd-hab"
    scene_dataset_dir: str

    # List of locations where scenes can be referenced by this config.
    # Used to resolve incomplete scene paths.
    # > "data/scene_datasets/hssd-hab/scenes"
    scene_directories: List[str] = []

    # Maps of template_name to render_asset path.
    # > Key: "2a0a80bf0b1b247799ed3a49bfdc9f9bf4fcf2b3"
    # > Value: "data/scene_datasets/hssd-hab/objects/2/2a0a80bf0b1b247799ed3a49bfdc9f9bf4fcf2b3.glb"
    objects: Dict[str, List[str]] = {}
    articulated_objects: Dict[str, List[str]] = {}
    stages: Dict[str, List[str]] = {}

    def __init__(self, scene_dataset_config_path: str):
        assert file_is_scene_dataset_config(scene_dataset_config_path)
        self.scene_dataset_dir = str(Path(scene_dataset_config_path).parent)
        scene_dataset_config = load_json(scene_dataset_config_path)

        # Find objects.
        if "objects" in scene_dataset_config:
            self.objects = search_dataset_render_assets(
                self.scene_dataset_dir,
                scene_dataset_config["objects"]["paths"][".json"],
            )

        # Find articulated objects.
        if "articulated_objects" in scene_dataset_config:
            self.articulated_objects = search_dataset_render_assets(
                self.scene_dataset_dir,
                scene_dataset_config["articulated_objects"]["paths"][".json"],
            )

        # Find stages.
        if "stages" in scene_dataset_config:
            self.stages = search_dataset_render_assets(
                self.scene_dataset_dir,
                scene_dataset_config["stages"]["paths"][".json"],
            )

        # Find scene path.
        if "scene_instances" in scene_dataset_config:
            scene_path_search_patterns = scene_dataset_config[
                "scene_instances"
            ]["paths"][".json"]
            for scene_path_search_pattern in scene_path_search_patterns:
                scene_path = Path.joinpath(
                    Path(self.scene_dataset_dir),
                    Path(scene_path_search_pattern),
                )
                # Remove search patterns (e.g. /*)
                while not scene_path.is_dir():
                    scene_path = scene_path.parent
                self.scene_directories.append(str(scene_path))

    def resolve_object(self, template_name: str) -> List[str]:
        stem = Path(template_name).stem.split(".")[0]
        return self.objects[stem]

    def resolve_articulated_object(self, template_name: str) -> List[str]:
        stem = Path(template_name).stem.split(".")[0]
        return self.articulated_objects[stem]

    def resolve_stage(self, template_name: str) -> List[str]:
        stem = Path(template_name).stem.split(".")[0]
        return self.stages[stem]


class ObjectDataset:
    """
    Represents a dataset that does not include a scene_dataset_config.json file.
    Typically, this is a folder referenced from an episode's "additional_obj_config_paths" field.
    These special datasets only contain regular objects - no stage or articulated objects.
    """

    render_assets: Set[str] = set()
    stem_to_resolved_path: Dict[str, str] = {}

    def __init__(self, dataset_path: str):
        assert Path(dataset_path).is_dir()
        object_config_json_paths = find_files(
            dataset_path, file_is_object_config
        )
        object_config_cache: Set[str] = set()
        for object_config_json_path in object_config_json_paths:
            # Ignore object if already found.
            if object_config_json_path in object_config_cache:
                continue
            object_config_cache.add(object_config_json_path)
            object_config = load_json(object_config_json_path)
            # > "2a0a80bf0b1b247799ed3a49bfdc9f9bf4fcf2b3.glb"
            object_render_asset_file_name = object_config["render_asset"]
            # > "data/scene_datasets/hssd-hab/objects/2"
            object_render_asset_dir = Path(object_config_json_path).parent
            # > "data/scene_datasets/hssd-hab/objects/2/2a0a80bf0b1b247799ed3a49bfdc9f9bf4fcf2b3.glb"
            object_render_asset_path = os.path.join(
                object_render_asset_dir, object_render_asset_file_name
            )
            resolved_path = resolve_relative_path(object_render_asset_path)
            self.render_assets.add(
                resolve_relative_path(resolved_path)
            )
            # > "2a0a80bf0b1b247799ed3a49bfdc9f9bf4fcf2b3"
            stem = Path(object_config_json_path).stem.split(".")[0]
            self.stem_to_resolved_path[stem] = resolved_path


class SceneInstance:
    """
    Upon construction, loads a scene_instance.json file and exposes its content.
    """

    group_name: str
    stage_render_asset: str
    object_render_assets: List[str]
    articulated_object_render_assets: List[str]

    def __init__(
        self,
        group_name: str,
        scene_instance_config_path: str,
        scene_dataset: SceneDataset,
    ) -> None:
        assert file_is_scene_config(scene_instance_config_path)
        self.group_name = group_name
        scene_instance_config = load_json(scene_instance_config_path)

        template_name = scene_instance_config["stage_instance"][
            "template_name"
        ]
        self.stage_render_asset = scene_dataset.resolve_stage(template_name)[0]

        object_render_assets = set()
        for instance in scene_instance_config["object_instances"]:
            template_name = instance["template_name"]
            assets = scene_dataset.resolve_object(template_name)
            for asset in assets:
                object_render_assets.add(asset)
        self.object_render_assets = list(object_render_assets)

        articulated_object_render_assets = set()
        for instance in scene_instance_config["articulated_object_instances"]:
            template_name = instance["template_name"]
            assets = scene_dataset.resolve_articulated_object(template_name)
            for asset in assets:
                articulated_object_render_assets.add(asset)
        self.articulated_object_render_assets = list(
            articulated_object_render_assets
        )


@dataclass
class AssetInfo:
    """
    Asset and the groups that contain it.
    Path is complete and relative to repository root (data/asset/blob.glb).
    """

    asset_path: str
    groups: List[str]

    def __init__(self, asset_path: str, groups: List[str]):
        self.asset_path = asset_path
        self.groups = groups


class GroupedSceneAssets:
    """
    All assets in the scenes along with their groups.
    Paths are complete and relative to repository root (data/asset/blob.glb).
    """

    stages: List[AssetInfo] = []
    objects: List[AssetInfo] = []
    articulated_objects: List[AssetInfo] = []

    def __init__(self, scene_instances: List[SceneInstance]):
        stage_groups: Dict[str, List[str]] = {}
        object_groups: Dict[str, List[str]] = {}
        articulated_object_groups: Dict[str, List[str]] = {}

        def add_group_reference(
            asset_path: str, group_name: str, groups: Dict[str, List[str]]
        ) -> None:
            if asset_path not in groups:
                groups[asset_path] = []
            groups[asset_path].append(group_name)

        for scene_instance in scene_instances:
            add_group_reference(
                scene_instance.stage_render_asset,
                scene_instance.group_name,
                stage_groups,
            )
            for obj in scene_instance.object_render_assets:
                add_group_reference(
                    obj, scene_instance.group_name, object_groups
                )
            for art_obj in scene_instance.articulated_object_render_assets:
                add_group_reference(
                    art_obj,
                    scene_instance.group_name,
                    articulated_object_groups,
                )

        for asset_path, groups in stage_groups.items():
            self.stages.append(AssetInfo(asset_path, groups))
        for asset_path, groups in object_groups.items():
            self.objects.append(AssetInfo(asset_path, groups))
        for asset_path, groups in articulated_object_groups.items():
            self.articulated_objects.append(AssetInfo(asset_path, groups))


#class EpisodeObjectAssets:
#    """
#    All objects used in a episode.
#    """
#    def __init__(self, assets: List[str], package_number: int, scene_dataset: SceneDataset):
#        for asset in assets:
#            scene_objects = scene_dataset.resolve_object(asset)
#            for scene_object in scene_objects:
                

class EpisodeSet:
    """
    Loads an episode set (.json.gz) and exposes its assets.
    """

    grouped_scene_assets: GroupedSceneAssets
    additional_datasets: Dict[str, ObjectDataset] = {}
    episodes: List[Dict[str, Any]] = []
    scene_datasets: Dict[str, SceneDataset] = {}

    def __init__(self, episodes_path: str):
        assert file_is_episode_set(
            episodes_path
        ), "Episodes must be supplied as a '.json.gz' file."
        with gzip.open(episodes_path, "rb") as file:
            json_data = file.read().decode("utf-8")
            loaded_data: Dict[str, List[Dict[str, Any]]] = json.loads(
                json_data
            )
        episodes = loaded_data["episodes"]
        self.episodes = episodes

        scene_datasets: Dict[str, SceneDataset] = {}
        scene_instances: Dict[str, SceneInstance] = {}
        additional_dataset_paths: Set[str] = set()
        for episode in episodes:
            # > "data/scene_datasets/hssd-hab/hssd-hab-uncluttered.scene_dataset_config.json"
            scene_dataset_config_path = episode["scene_dataset_config"]
            # Load dataset.
            if scene_dataset_config_path not in scene_datasets:
                scene_datasets[scene_dataset_config_path] = SceneDataset(
                    str(scene_dataset_config_path)
                )
            scene_dataset = scene_datasets[scene_dataset_config_path]
            # > "data/scene_datasets/hssd-hab/scenes-uncluttered/102344193.scene_instance.json"
            # BEWARE: "scene_id" can either be a path or a file stem (without extension).
            scene_instance_config_path = episode["scene_id"]
            if not file_is_scene_config(scene_instance_config_path):
                for (
                    dataset_scene_search_path
                ) in scene_dataset.scene_directories:
                    # Try to resolve the incomplete scene file name.
                    scene_file_candidate = Path(
                        os.path.join(
                            dataset_scene_search_path,
                            scene_instance_config_path
                            + ".scene_instance.json",
                        )
                    )
                    if scene_file_candidate.is_file():
                        scene_instance_config_path = str(scene_file_candidate)
                        break
            assert file_is_scene_config(
                scene_instance_config_path
            ), "Unsupported dataset definition."
            # Load scene instance.
            scene_instance_config_path = resolve_relative_path(
                scene_instance_config_path
            )
            if scene_instance_config_path not in scene_instances:
                scene_instances[scene_instance_config_path] = SceneInstance(
                    group_name=scene_instance_config_path,
                    scene_instance_config_path=str(scene_instance_config_path),
                    scene_dataset=scene_dataset,
                )
            # Find additional datasets referenced by the episode.
            if "additional_obj_config_paths" in episode:
                for additional_dataset_path in episode[
                    "additional_obj_config_paths"
                ]:
                    # > "data/objects/ycb/configs/"
                    additional_dataset_path = resolve_relative_path(
                        additional_dataset_path
                    )
                    additional_dataset_paths.add(str(additional_dataset_path))

        # Load additional datasets (e.g. ycb).
        for additional_dataset_path in additional_dataset_paths:
            self.additional_datasets[additional_dataset_path] = ObjectDataset(
                str(additional_dataset_path)
            )

        # Group assets per scene dependencies.
        self.grouped_scene_assets = GroupedSceneAssets(
            list(scene_instances.values())
        )

        self.scene_datasets = scene_datasets


def create_metadata_file(results: List[Dict[str, Any]]):
    """
    Create output metadata file.
    This file is consumed by custom external asset pipelines (e.g. Unity) to manage assets.

    Contents:
    * version: Version of the file format. Bump when breaking backward compatibility.
    * local_group_name: Name of the local group. See 'LOCAL_GROUP_NAME'.
    * groups: List of groups and their assets.
              Groups are used to determine how to package remote assets.
    """
    # Aggregate groups from processing results.
    groups: Dict[str, List[str]] = {}
    for result in results:
        if (
            Path(result["dest_path"]).is_file()
            and result["status"] != "error"
            and "groups" in result
        ):
            for group in result["groups"]:
                file_rel_path = result["dest_path"].removeprefix(OUTPUT_DIR)
                if group not in groups:
                    groups[group] = []
                groups[group].append(file_rel_path)

    # Compose file content.
    content: Dict[str, Any] = {}
    content["version"] = METADATA_FILE_VERSION
    content["local_group_name"] = LOCAL_GROUP_NAME
    content["groups"] = groups

    # Save file.
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_METADATA_FILE_NAME)
    with open(output_path, "w") as f:
        json.dump(content, f, ensure_ascii=False)


def process_model(args):
    job, counter, lock, total_models, verbose = args

    result = {
        "source_path": job.source_path,
        "dest_path": job.dest_path,
        "groups": job.groups,
    }

    if os.path.isfile(job.dest_path):
        if verbose:
            print(f"Skipping:   {job.source_path}")
        result["status"] = "skipped"
        return result

    if verbose:
        print(f"Processing: {job.source_path}.")

    # Create all necessary subdirectories
    os.makedirs(os.path.dirname(job.dest_path), exist_ok=True)

    if Path(job.dest_path).suffix == ".obj":
        shutil.copyfile(job.source_path, job.dest_path, follow_symlinks=True)
        result["status"] = "copied"
        return result
    job_type: JobType = job.job_type    
    if job_type == JobType.COPY:
        shutil.copyfile(job.source_path, job.dest_path, follow_symlinks=True)
        result["status"] = "copied"
        return result
    else:
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
                decimate.close()
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
                decimate.close()
                print(
                    f"Unable to decimate: {job.source_path}. Copying as-is to output."
                )
                shutil.copyfile(
                    job.source_path, job.dest_path, follow_symlinks=True
                )
                result["status"] = "error"
                return result

    if job_type == JobType.PROCESS and job.simplify and verbose:
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

    with lock:
        counter.value += 1
        if verbose:
            print(
                f"{counter.value} out of {total_models} models have been processed so far."
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
    total_copied = 0

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

    # Create the output metadata file.
    create_metadata_file(results)

    elapsed_time = time.time() - start_time
    print(f"Elapsed time (s): {elapsed_time}")
    if not config.use_multiprocessing:
        print("Add --use-multiprocessing to speed-up processing.")


def load_json(path: str) -> Dict[str, Any]:
    with open(path) as file:
        output = json.load(file)
        return output


def main():
    parser = argparse.ArgumentParser(
        description="Get all render asset files associated with a given episode set."
    )
    parser.add_argument(
        "--episodes",
        type=str,
        help="Episodes (.json.gz).",
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

    # Load episodes.
    assert Path(args.episodes).is_file()
    episode_set = EpisodeSet(args.episodes)

    jobs: List[Job] = []

    # Add additional datasets.
    additional_asset_group_size = 50
    objects_in_current_group = 0
    current_group_index = 0
    #shared_scene_objects: Set[str] = set()  # Objects that are both in scenes and episodes
    processed_objects: Set[str] = set()
    for episode in episode_set.episodes:
        rigid_objs = episode["rigid_objs"]
        for rigid_obj in rigid_objs:
            resolved_rigid_objs: List[str] = []
            rigid_obj_stem = Path(rigid_obj[0]).stem.split(".")[0]
            for object_dataset in episode_set.additional_datasets.values():
                if rigid_obj_stem in object_dataset.stem_to_resolved_path:
                    resolved_rigid_objs = [object_dataset.stem_to_resolved_path[rigid_obj_stem]]
                    continue
            # Look for the object in the scene dataset.
            # HACK: We create duplicates for these objects.
            #for scene_dataset in episode_set.scene_datasets.values():
            #    if rigid_obj_stem in scene_dataset.objects:
            #        resolved_rigid_objs = scene_dataset.objects[rigid_obj_stem]
            #        for resolved in resolved_rigid_objs:
            #            shared_scene_objects.add(resolved)
            #        continue
            assert len(resolved_rigid_objs) > 0
            for resolved_rigid_obj in resolved_rigid_objs:
                jobs.append(
                    Job(
                        asset_path=resolved_rigid_obj,
                        output_dir=OUTPUT_DIR,
                        groups=[str(current_group_index)],
                        simplify=False,
                    )
                )
                processed_objects.add(resolved_rigid_obj)
            objects_in_current_group += 1
        # Note: additional_asset_group_size can be exceeded to allow packaging episode assets together
        if objects_in_current_group >= additional_asset_group_size:
            objects_in_current_group -= additional_asset_group_size
            current_group_index += 1
    ## Sloppy: Add missing assets.
    #for dataset in episode_set.additional_datasets.values():
    #    for asset_path in dataset.render_assets:
    #        jobs.append(
    #            Job(
    #                asset_path=asset_path,
    #                output_dir=OUTPUT_DIR,
    #                groups=[str(current_group_index)],
    #                simplify=False,
    #            )
    #        )
    #        objects_in_current_group += 1
    #        if objects_in_current_group >= additional_asset_group_size:
    #            objects_in_current_group -= additional_asset_group_size
    #            current_group_index += 1

    # Add scene stages.
    # Grouped by scenes.
    for obj in episode_set.grouped_scene_assets.stages:
        jobs.append(
            Job(
                asset_path=obj.asset_path,
                output_dir=OUTPUT_DIR,
                groups=obj.groups,
                simplify=False,
            )
        )

    # Add scene objects.
    # Grouped by scenes, excluding objects also in episodes.
    for obj in episode_set.grouped_scene_assets.objects:
        #if obj.asset_path in shared_scene_objects:
        #    continue
        jobs.append(
            Job(
                asset_path=obj.asset_path,
                output_dir=OUTPUT_DIR,
                groups=obj.groups,
                simplify=True,
            )
        )

    # Add articulated objects.
    # Grouped by scenes.
    for obj in episode_set.grouped_scene_assets.articulated_objects:
        jobs.append(
            Job(
                asset_path=obj.asset_path,
                output_dir=OUTPUT_DIR,
                groups=obj.groups,
                simplify=False,
            )
        )

    # Add spot models
    for filename in Path("data/robots/hab_spot_arm/meshesColored").rglob(
        "*.glb"
    ):
        jobs.append(
            Job(
                asset_path=str(filename),
                output_dir=OUTPUT_DIR,
                groups=[],
                simplify=False,
            )
        )

    # Add humanoid models
    for filename in Path("data/humanoids/humanoid_data").rglob(
        "*.glb"
    ):
        jobs.append(
            Job(
                asset_path=str(filename),
                output_dir=OUTPUT_DIR,
                groups=[],
                simplify=False,
                job_type=JobType.COPY,
            )
        )

    # Verify jobs.
    verify_jobs(jobs, OUTPUT_DIR)

    #groups = {}
    #for job in jobs:
    #    for group in job.groups:
    #        if group not in groups:
    #            groups[group] = 0
    #        groups[group] += 1
    #for group, count in groups.items():
    #    print(f"{group}: {str(count)}")

    # Start processing.
    simplify_models(jobs, config)


if __name__ == "__main__":
    main()
