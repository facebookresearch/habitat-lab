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
from typing import Any, Callable, Dict, List, Set, Tuple
import gzip

import decimate

OUTPUT_DIR = "data/_processing_output/"
OMIT_BLACK_LIST = False
OMIT_GRAY_LIST = False
PROCESS_COUNT = os.cpu_count()


class Job:
    def __init__(self, asset_path: str, output_dir: str, groups: List[str], simplify: bool):
        self.source_path = asset_path
        self.dest_path = os.path.join(output_dir, asset_path)
        self.groups = groups
        self.simplify = simplify


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
    return (filepath.endswith(".object_config.json") or
            filepath.endswith(".ao_config.json") or
            filepath.endswith(".stage_config.json"))

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


def file_is_glb(filepath: str) -> bool:
    """
    Return whether or not the file is a glb.
    """
    return filepath.endswith(".glb")


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
        assert Path(job.source_path).is_file(), f"Source path is not a file: '{job.source_path}'."
        # Check that all source paths start with "data/".
        prefix = "data/"
        prefix_length = len(prefix)
        assert str(job.source_path)[0:prefix_length] == prefix, f"Source path not in '{prefix}': '{job.source_path}'."
        # Check that all dest paths start with "{output_path}/data/"
        prefix = os.path.join(output_path, "data/")
        prefix_length = len(prefix)
        assert str(job.dest_path)[0:prefix_length] == prefix, f"Dest path not in '{prefix}': '{job.dest_path}'."
        # Check that all job paths are unique.
        assert not job.source_path in source_set, f"Duplicate source asset: '{job.source_path}'."
        assert not job.dest_path in dest_set, f"Duplicate destination asset: '{job.dest_path}'."
        source_set.add(job.source_path)
        dest_set.add(job.dest_path)

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



def search_dataset_render_assets(scene_dataset_dir: str, directory_search_patterns: List[str]) -> Dict[str, str]:
    """
    scene_dataset_dir: "data/scene_datasets/hssd-hab"
    directory_search_patterns: ["objects/*", "objects/decomposed/*"]
    """
    output: Dict[str, str] = {}
    object_config_json_path_cache: set[str] = set()

    for directory_search_pattern in directory_search_patterns:
        # > "data/scene_datasets/hssd-hab/objects/*"
        object_config_search_pattern = os.path.join(scene_dataset_dir, directory_search_pattern)
        # > "data/scene_datasets/hssd-hab/objects"
        object_config_search_dir = Path(object_config_search_pattern)
        while not Path.is_dir(object_config_search_dir):
            object_config_search_dir = object_config_search_dir.parent
        # Find all object configs in the specified path
        # > "data/scene_datasets/hssd-hab/objects/2/2a0a80bf0b1b247799ed3a49bfdc9f9bf4fcf2b3.object_config.json"
        object_config_json_paths = find_files(object_config_search_dir, file_is_render_asset_config)
        for object_config_json_path in object_config_json_paths:
            # Ignore object if already found.
            if object_config_json_path in object_config_json_path_cache:
                continue
            object_config_json_path_cache.add(object_config_json_path)
            object_config = load_json(object_config_json_path)
            # > "2a0a80bf0b1b247799ed3a49bfdc9f9bf4fcf2b3.glb"
            object_render_asset_file_name = object_config["render_asset"]
            # > "2a0a80bf0b1b247799ed3a49bfdc9f9bf4fcf2b3
            object_template_name = Path(object_render_asset_file_name).stem
            # > "data/scene_datasets/hssd-hab/objects/2"
            object_render_asset_dir = Path(object_config_json_path).parent
            # > "data/scene_datasets/hssd-hab/objects/2/2a0a80bf0b1b247799ed3a49bfdc9f9bf4fcf2b3.glb"
            object_render_asset_path = os.path.join(object_render_asset_dir, object_render_asset_file_name)
            output[object_template_name] = object_render_asset_path

    return output


class SceneDataset:
    """
    Upon construction, loads a scene_dataset_config.json file and exposes its content.
    """
    # Location of the scene_dataset_config.json.
    # > "data/scene_datasets/hssd-hab"
    scene_dataset_dir: str

    # Maps of template_name to render_asset path.
    # > Key: "2a0a80bf0b1b247799ed3a49bfdc9f9bf4fcf2b3"
    # > Value: "data/scene_datasets/hssd-hab/objects/2/2a0a80bf0b1b247799ed3a49bfdc9f9bf4fcf2b3.glb"
    objects: Dict[str, str] = {}
    articulated_objects: Dict[str, str] = {}
    stages: Dict[str, str] = {}

    def __init__(self, scene_dataset_config_path: str):
        self.scene_dataset_dir: str = Path(scene_dataset_config_path).parent
        scene_dataset_config = load_json(scene_dataset_config_path)

        # Find objects.
        if "objects" in scene_dataset_config:
            self.objects = search_dataset_render_assets(self.scene_dataset_dir,
                                                scene_dataset_config["objects"]["paths"][".json"])

        # Find articulated objects.
        if "articulated_objects" in scene_dataset_config:
            self.articulated_objects = search_dataset_render_assets(self.scene_dataset_dir,
                                                            scene_dataset_config["articulated_objects"]["paths"][".json"])

        # Find stages.
        if "stages" in scene_dataset_config:
            self.stages = search_dataset_render_assets(self.scene_dataset_dir,
                                            scene_dataset_config["stages"]["paths"][".json"])
    
    def resolve_object(self, template_name: str) -> str:
        stem = Path(template_name).stem
        return self.objects[stem]
    
    def resolve_articulated_object(self, template_name: str) -> str:
        stem = Path(template_name).stem
        return self.articulated_objects[stem]
    
    def resolve_stage(self, template_name: str) -> str:
        stem = Path(template_name).stem
        return self.stages[stem]


class ObjectDataset:
    """
    Represents a dataset that does not include a scene_dataset_config.json file.
    Typically, this is a folder referenced from an episode's "additional_obj_config_paths" field.
    These special datasets only contain regular objects - no stage or articulated objects.
    """
    render_assets: List[str] = []

    def __init__(self, dataset_path: str):
        assert Path(dataset_path).is_dir()
        object_config_json_paths = find_files(dataset_path, file_is_object_config)
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
            object_render_asset_path = os.path.join(object_render_asset_dir, object_render_asset_file_name)
            self.render_assets.append(object_render_asset_path)

class SceneInstance:
    """
    Upon construction, loads a scene_instance.json file and exposes its content.
    """
    group_name: str
    stage_render_asset: str
    object_render_assets: List[str]
    articulated_object_render_assets: List[str]

    def __init__(self, group_name: str, scene_instance_config_path: str, scene_dataset: SceneDataset) -> None:
        self.group_name = group_name
        scene_instance_config = load_json(scene_instance_config_path)

        template_name = scene_instance_config["stage_instance"]["template_name"]
        self.stage_render_asset = scene_dataset.resolve_stage(template_name)

        object_render_assets = set()
        for instance in scene_instance_config["object_instances"]:
            template_name = instance["template_name"]
            object_render_assets.add(scene_dataset.resolve_object(template_name))
        self.object_render_assets = list(object_render_assets)

        articulated_object_render_assets = set()
        for instance in scene_instance_config["articulated_object_instances"]:
            template_name = instance["template_name"]
            articulated_object_render_assets.add(scene_dataset.resolve_articulated_object(template_name))
        self.articulated_object_render_assets = list(articulated_object_render_assets)




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

        def add_group_reference(asset_path: str, group_name: str, groups: Dict[str, List[str]]) -> None:
            if asset_path not in groups:
                groups[asset_path] = []
            groups[asset_path].append(group_name)

        for scene_instance in scene_instances:
            add_group_reference(
                scene_instance.stage_render_asset,
                scene_instance.group_name,
                stage_groups
            )
            for obj in scene_instance.object_render_assets:
                add_group_reference(
                    obj,
                    scene_instance.group_name,
                    object_groups
                )
            for art_obj in scene_instance.articulated_object_render_assets:
                add_group_reference(
                    art_obj,
                    scene_instance.group_name,
                    articulated_object_groups
                )

        for asset_path, groups in stage_groups.items():
            self.stages.append(AssetInfo(asset_path, groups))
        for asset_path, groups in object_groups.items():
            self.objects.append(AssetInfo(asset_path, groups))
        for asset_path, groups in articulated_object_groups.items():
            self.articulated_objects.append(AssetInfo(asset_path, groups))
        
        
class EpisodeSet:
    """
    Loads and episode set and exposes its assets.
    """
    grouped_scene_assets: GroupedSceneAssets
    additional_datasets: Dict[str, ObjectDataset] = {}

    def __init__(self, episodes_path: str):
        assert file_is_episode_set(episodes_path), "Episodes must be supplied as a '.json.gz' file."
        with gzip.open(episodes_path, "rb") as file:
            json_data = file.read().decode("utf-8")
            loaded_data = json.loads(json_data)
        episodes: List[Dict[Any]] = loaded_data["episodes"]

        scene_datasets: Dict[str, SceneDataset] = {}
        scene_instances: Dict[str, SceneInstance] = {}
        additional_dataset_paths: Set[str] = set()
        for episode in episodes:
            # > "data/scene_datasets/hssd-hab/hssd-hab-uncluttered.scene_dataset_config.json"
            scene_dataset_config_path = episode["scene_dataset_config"]
            # Load dataset.
            if scene_dataset_config_path not in scene_datasets:
                scene_datasets[scene_dataset_config_path] = SceneDataset(scene_dataset_config_path)
            scene_dataset = scene_datasets[scene_dataset_config_path]
            # > "data/scene_datasets/hssd-hab/scenes-uncluttered/102344193.scene_instance.json"
            scene_instance_config_path = episode["scene_id"]
            # Load scene instance.
            if scene_instance_config_path not in scene_instances:
                scene_instances[scene_instance_config_path] = SceneInstance(
                    group_name=scene_instance_config_path,
                    scene_instance_config_path=scene_instance_config_path,
                    scene_dataset=scene_dataset)
            # Find additional datasets referenced by the episode.
            if "additional_obj_config_paths" in episode:
                for additional_dataset_path in episode["additional_obj_config_paths"]:
                    # > "data/objects/ycb/configs/"
                    additional_dataset_paths.add(additional_dataset_path)

        # Load additional datasets (e.g. ycb).
        for additional_dataset_path in additional_dataset_paths:
            self.additional_datasets[additional_dataset_path] = ObjectDataset(additional_dataset_path)

        # Group assets per scene dependencies.            
        self.grouped_scene_assets = GroupedSceneAssets(scene_instances.values())



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
        "dest_path": job.dest_path,
        "status": "ok",
        "groups": job.groups
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

    # Create the dataset.json file containing all dependencies.
    groups: Dict[str, List[str]] = {}
    for result in results:
        if result["status"] == "ok" and "groups" in result:
            pkgs = result["groups"]
            for pkg in pkgs:
                file_rel_path = result["dest_path"].removeprefix(OUTPUT_DIR)
                if pkg not in groups:
                    groups[pkg] = []
                groups[pkg].append(file_rel_path)
    groups_json_path = os.path.join(OUTPUT_DIR, "dataset.json")
    with open(groups_json_path, 'w') as f:
        json.dump(groups, f, ensure_ascii=False)

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
        description="Get all .glb render asset files associated with a given scene."
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

    # TODO: Parameterize
    episodes_raw = "data/episodes/rearrange_ep_dataset.json.gz"

    # Load episodes.
    episode_set = EpisodeSet(episodes_raw)

    jobs: List[Job] = []

    # Add scene stages.
    # Grouped by scenes.
    for obj in episode_set.grouped_scene_assets.stages:
        jobs.append(Job(
            asset_path=obj.asset_path,
            output_dir=OUTPUT_DIR,
            groups=obj.groups,
            simplify=False,
        ))

    # Add scene objects.
    # Grouped by scenes.
    for obj in episode_set.grouped_scene_assets.objects:
        jobs.append(Job(
            asset_path=obj.asset_path,
            output_dir=OUTPUT_DIR,
            groups=obj.groups,
            simplify=True,
        ))

    # Add articulated objects.
    # Grouped by scenes.
    for obj in episode_set.grouped_scene_assets.articulated_objects:
        jobs.append(Job(
            asset_path=obj.asset_path,
            output_dir=OUTPUT_DIR,
            groups=obj.groups,
            simplify=False,
        ))

    # Add additional datasets.
    for dataset in episode_set.additional_datasets.values():
        for asset_path in dataset.render_assets:
            jobs.append(Job(
                asset_path=asset_path,
                output_dir=OUTPUT_DIR,
                groups=[],
                simplify=False,
            ))

    # Add spot models
    for filename in Path("data/robots/hab_spot_arm/meshesColored").rglob(
        "*.glb"
    ):
        jobs.append(Job(
            asset_path=filename,
            output_dir=OUTPUT_DIR,
            groups=[],
            simplify=False,
        ))

    # Verify jobs.
    verify_jobs(jobs, OUTPUT_DIR)
    
    # Start processing.
    simplify_models(jobs, config)


if __name__ == "__main__":
    main()