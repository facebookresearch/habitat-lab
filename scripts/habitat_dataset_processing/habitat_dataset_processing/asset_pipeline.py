#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import fnmatch
import json
import os
import shutil
from pathlib import Path
from typing import Any, Optional

from habitat_dataset_processing.asset_processor import (
    create_metadata_file,
    process_models,
)
from habitat_dataset_processing.configs import (
    AssetSource,
    Config,
    GroupType,
    HabitatDatasetSource,
    ProcessingSettings,
)
from habitat_dataset_processing.job import Job
from habitat_dataset_processing.util import (
    get_dependencies,
    resolve_relative_path,
    resolve_relative_path_with_wildcard,
)


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


def find_files_with_extension(directory: str, extension: str) -> list[str]:
    matches: list[str] = []
    for root, _dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, f"*.{extension}"):
            matches.append(os.path.join(root, filename))
    return matches


def remove_duplicates(items: list[str]) -> list[str]:
    output: set[str] = set()
    for item in items:
        output.add(item)
    return list(output)


def load_json(path: str) -> Optional[dict[str, Any]]:
    try:
        with open(path, "r") as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Could not load JSON file: '{path}'.")
    except Exception as e:
        print(f"The specified JSON file is invalid: '{path}'.\n{e}")
    return None


def absolute_file_paths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


def get_template_name(config_path: str) -> str:
    """Get the template name of an object. The template name is equivalent to the filename before the extension."""
    return Path(config_path).stem.split(".")[0]


class Asset:
    def __init__(
        self,
        path: str,
        settings: ProcessingSettings,
    ):
        self.path = resolve_relative_path(path)
        self.settings = settings
        self.scenes: set[str] = set()
        self.dependencies = get_dependencies(self.path)


class AssetDatabase:
    """
    Set of all assets in a Habitat dataset.

    Helps track Habitat-specific IDs like template names.
    """

    def __init__(self, database_name: str):
        self._unresolved_assets: set[str] = set()
        self._template_name_to_asset: dict[str, Asset] = {}
        self._database_name = database_name

    def register_asset(
        self,
        asset_template_name: str,
        asset_path: str,
        settings: ProcessingSettings,
    ):
        """Add an asset to the collection."""
        assert os.path.exists(
            asset_path
        ), f"Could not find asset path: {asset_path}. Aborting."
        self._template_name_to_asset[asset_template_name] = Asset(
            path=asset_path,
            settings=settings,
        )

    def register_asset_from_config(
        self, config_path: str, settings: ProcessingSettings
    ):
        data = load_json(config_path)
        assert (
            data is not None
        ), f"Invalid object or stage config: '{config_path}'."
        object_dir: str = str(Path(config_path).parent)
        render_asset_file_name: Optional[str] = None
        if "render_asset" in data and data["render_asset"] != "":
            render_asset_file_name = data["render_asset"]
        elif "urdf_filepath" in data and data["urdf_filepath"] != "":
            render_asset_file_name = data["urdf_filepath"]
        render_asset_path = os.path.join(object_dir, render_asset_file_name)
        template_name = get_template_name(config_path)
        self.register_asset(template_name, render_asset_path, settings)

    def find_asset_by_template_name(
        self, asset_template_name: str
    ) -> Optional[Asset]:
        """Resolve a 'template name' into a list of render asset paths."""
        if asset_template_name not in self._template_name_to_asset:
            self._unresolved_assets.add(asset_template_name)
        asset = self._template_name_to_asset.get(asset_template_name, None)
        return asset

    def remove_orphan_assets(self):
        """Remove all assets that don't appear in any scene."""
        assets_to_remove: list[str] = []
        for template_name, asset in self._template_name_to_asset.items():
            if len(asset.scenes) == 0:
                assets_to_remove.append(template_name)
        for asset_to_remove in assets_to_remove:
            del self._template_name_to_asset[asset_to_remove]

    @property
    def database_name(self) -> str:
        return self._database_name


class AssetPipeline:
    def __init__(
        self,
        datasets: list[HabitatDatasetSource],
        additional_assets: list[AssetSource],
        output_subdir: str = "",
    ):
        self._output_subdir = output_subdir

        # Parse CLI arguments.
        parser = argparse.ArgumentParser(
            description="Process Habitat datasets for usage in external engines.",
        )
        parser.add_argument(
            "--input",
            type=str,
            help="Path of the `data/` directory to process.",
        )
        parser.add_argument(
            "--output",
            type=str,
            default="_data_processing_output",
            help="Output directory. Will be created if it doesn't exist.",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            default=False,
            help="Increase logging verbosity.",
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            default=False,
            help="Disable multiprocessing to allow for debugging.",
        )
        args = parser.parse_args()

        assert os.path.exists(
            args.input
        ), f"The specified `--data` directory does not exist: `{args.data}`."

        self._config = Config(
            input_dir=args.input,
            data_folder_name=Path(args.input).name,
            output_dir=args.output,
            verbose=args.verbose,
            use_multiprocessing=not args.debug,
            datasets=datasets,
            additional_assets=additional_assets,
        )

        self._databases: list[AssetDatabase] = []

    def _load_habitat_dataset(self, source: HabitatDatasetSource):
        dataset_config_path = os.path.join(
            self._config.input_dir, source.dataset_config
        )
        data = load_json(dataset_config_path)
        assert (
            data is not None
        ), f"Invalid dataset config: {dataset_config_path}"
        database = AssetDatabase(source.name)

        whitelist = source.scene_whitelist
        blacklist = source.scene_blacklist
        include_orphan_assets = source.include_orphan_assets

        # Find all asset directories.
        def get_asset_dirs(asset_dirs: str) -> list[str]:
            output: list[str] = []
            for asset_dir in asset_dirs:
                new_dir = os.path.join(dataset_dir, asset_dir)
                output.append(resolve_relative_path_with_wildcard(new_dir))
            return output

        # Find all configs.
        def get_configs(config_dirs: list[str], file_extension: str):
            configs: set[str] = set()
            for config_dir in config_dirs:
                files = find_files_with_extension(config_dir, file_extension)
                for file in files:
                    configs.add(resolve_relative_path(file))
            return list(configs)

        dataset_dir: str = str(Path(dataset_config_path).parent)
        if "stages" in data:
            dirs = get_asset_dirs(data["stages"]["paths"][".json"])
            paths = get_configs(dirs, "stage_config.json")
            for path in paths:
                database.register_asset_from_config(path, source.stages)
        if "objects" in data:
            dirs = get_asset_dirs(data["objects"]["paths"][".json"])
            paths = get_configs(dirs, "object_config.json")
            for path in paths:
                database.register_asset_from_config(path, source.objects)
        if "articulated_objects" in data:
            dirs = get_asset_dirs(
                data["articulated_objects"]["paths"][".json"]
            )
            paths = get_configs(dirs, "ao_config.json")
            for path in paths:
                database.register_asset_from_config(
                    path, source.articulated_objects
                )
        if "scene_instances" in data:
            dirs = get_asset_dirs(data["scene_instances"]["paths"][".json"])
            scene_paths = get_configs(dirs, "scene_instance.json")

            # Find which scenes contain the assets for grouping.
            def get_all_template_names_in_scene(
                scene_config: str,
            ) -> list[str]:
                output: set[str] = set()
                data = load_json(scene_config)
                assert data is not None, "Invalid scene config."
                for object_instance in data.get("object_instances", []):
                    assert "template_name" in object_instance
                    template_name = object_instance["template_name"]
                    template_name = get_template_name(template_name)
                    output.add(template_name)
                for object_instance in data.get(
                    "articulated_object_instances", []
                ):
                    assert "template_name" in object_instance
                    template_name = object_instance["template_name"]
                    template_name = get_template_name(template_name)
                    output.add(template_name)
                if "stage_instance" in data:
                    assert "template_name" in data["stage_instance"]
                    # Note: The stage template names may sometimes contain a relative path.
                    template_name = data["stage_instance"]["template_name"]
                    if "/" in template_name:
                        template_name = template_name.split("/")[-1]
                    get_template_name(template_name)
                    output.add(template_name)
                return list(output)

            for scene_path in scene_paths:
                scene_name = get_template_name(scene_path)

                if whitelist is not None and scene_name not in whitelist:
                    continue
                if blacklist is not None and scene_name in blacklist:
                    continue

                template_names = get_all_template_names_in_scene(scene_path)
                for template_name in template_names:
                    asset = database.find_asset_by_template_name(template_name)
                    assert (
                        asset is not None
                    ), f"Template name {template_name} could not be resolved."
                    asset.scenes.add(scene_name)
        else:
            assert (
                whitelist is None
            ), f"A scene whitelist is specified for a dataset that does not contain any scene: '{source.dataset_config}'."
            assert (
                blacklist is None
            ), f"A scene blacklist is specified for a dataset that does not contain any scene: '{source.dataset_config}'."
            assert (
                include_orphan_assets == True
            ), f"`include_orphan_assets` must be set for datasets that does not contain any scene: '{source.dataset_config}'."

        # Remove assets that do not appear in any scene.
        if (
            not include_orphan_assets
            or whitelist is not None
            or blacklist is not None
        ):
            database.remove_orphan_assets()

        self._databases.append(database)

    def _load_assets(self, source: AssetSource):
        database_name = source.name
        database = AssetDatabase(database_name)

        def split_string_at_first_asterisk(s: str) -> tuple[str, str]:
            index = s.find("*")
            if index != -1:
                before_asterisk = s[:index]
                after_asterisk = s[index:]
                return before_asterisk, after_asterisk
            else:
                return s, "*"

        for asset_search_pattern in source.assets:
            asset_search_pattern = os.path.join(
                self._config.input_dir, asset_search_pattern
            )
            if "*" in asset_search_pattern:
                directory, glob = split_string_at_first_asterisk(
                    asset_search_pattern
                )
                asset_paths = [
                    str(path) for path in Path(directory).rglob(glob)
                ]
            else:
                asset_paths = [asset_search_pattern]
            for asset_path in asset_paths:
                assert os.path.exists(
                    asset_path
                ), f"Invalid asset path: '{asset_path}'."
                database.register_asset(
                    asset_path, asset_path, source.settings
                )

        self._databases.append(database)

    def process(self):
        config = self._config

        # Load datasets
        for dataset in config.datasets:
            self._load_habitat_dataset(dataset)

        # Load additional assets
        for assets in config.additional_assets:
            self._load_assets(assets)

        # (Re)create output dir
        output_dir = os.path.join(config.output_dir, self._output_subdir)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # Create a list of processing commands
        jobs: list[Job] = []
        groups: dict[str, list[str]] = {}
        local_group_name = "local"
        max_chunk_size = 10
        chunk_index = 0
        current_chunk_length = 0
        for database_index in range(len(self._databases)):
            database = self._databases[database_index]

            # Start a new chunk
            if current_chunk_length > 0:
                current_chunk_length = 0
                chunk_index += 1

            for asset in database._template_name_to_asset.values():
                for asset_path in asset.dependencies.union([asset.path]):
                    relative_path = os.path.join(
                        config.data_folder_name,
                        asset_path[len(config.input_dir) + 1 :],
                    )
                    dest_path = os.path.join(output_dir, relative_path)
                    jobs.append(
                        Job(
                            asset_path=asset_path,
                            dest_path=dest_path,
                            operation=asset.settings.operation,
                            simplify=asset.settings.decimate,
                        )
                    )

                    def add_asset_to_group(path: str, group: str):
                        if group not in groups:
                            groups[group] = []
                        groups[group].append(path)

                    # Compute groups
                    group_type = asset.settings.group

                    # LOCAL: Add to the single 'local' group.
                    if group_type == GroupType.LOCAL:
                        add_asset_to_group(relative_path, local_group_name)
                    # GROUP_BY_DATASET: Add to a per-database group.
                    elif group_type == GroupType.GROUP_BY_DATASET:
                        add_asset_to_group(
                            relative_path, database.database_name
                        )
                    # GROUP_BY_CHUNK: Add to a chunk.
                    elif group_type == GroupType.GROUP_BY_CHUNK:
                        add_asset_to_group(relative_path, str(chunk_index))
                        current_chunk_length += 1
                        if current_chunk_length > max_chunk_size:
                            current_chunk_length = 0
                            chunk_index += 1
                    # GROUP_BY_SCENE: Add to all scene groups containing the asset.
                    elif group_type == GroupType.GROUP_BY_SCENE:
                        if len(asset.scenes) > 0:
                            for scene in asset.scenes:
                                add_asset_to_group(relative_path, scene)
                        else:
                            # If the asset is orphan, fallback to 'GROUP_BY_CHUNK'.
                            add_asset_to_group(relative_path, str(chunk_index))
                            current_chunk_length += 1
                            if current_chunk_length > max_chunk_size:
                                current_chunk_length = 0
                                chunk_index += 1

        process_models(jobs, config)
        create_metadata_file(groups, output_dir)
