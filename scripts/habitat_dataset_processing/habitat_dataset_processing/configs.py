#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class GroupType(Enum):
    """
    Hints how the assets should be packaged.
    The assets in the generated 'assets.json' will be grouped according to this enum.
    This maybe used by the client to package some assets in remote bundles to reduce build size.
    """

    LOCAL = 0
    """
    The assets should be packaged along with the build. This increases the build size (WebGL app, Android APK, etc.).
    Use for assets that are expected to be used every time.
    """

    GROUP_BY_DATASET = 1
    """
    Group all assets within this data source together.
    Use when most assets within a dataset are intended to be used together (e.g. a robot).
    """

    GROUP_BY_CHUNK = 2
    """
    Assets will be grouped in arbitrary chunks.
    Use for large datasets with many objects that are occasionally used (e.g. ycb, ovmm, etc.).
    """

    GROUP_BY_SCENE = 3
    """
    Assets will be grouped by scenes.
    Use for scene datasets (e.g. hssd).
    """


class Operation(Enum):
    IGNORE = 0
    """Skip the asset."""

    COPY = 1
    """Copy the asset as-is, skipping all processing."""

    PROCESS = 2
    """Process the asset with the selected processor (only `magnum_decimation` is supported at the moment)."""


@dataclass
class ProcessingSettings:
    """
    List of assets to import. Uses 'glob' format.
    Examples:
    - "data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf"
    - "data/humanoids/humanoid_data/**/*.glb"
    """

    operation: Operation
    decimate: bool
    group: GroupType

    def __init__(
        self,
        operation: Operation,
        decimate: bool,
        group: GroupType = GroupType.LOCAL,
    ):
        self.operation = operation
        self.decimate = decimate
        self.group = group


@dataclass
class HabitatDatasetSource:
    """
    Defines how to import a Habitat dataset (`scene_dataset_config.json` file).
    These datasets can include stages, objects and/or articulated objects.
    """

    name: str
    """Name of the dataset."""

    dataset_config: str
    """Path to the `scene_dataset_config.json` file describing the dataset."""

    scene_whitelist: Optional[list[str]]
    """If present, defines the scenes to be included in the output."""

    scene_blacklist: Optional[list[str]]
    """If present, defines the scenes to be excluded from the output."""

    include_orphan_assets: bool

    stages: Optional[ProcessingSettings]
    objects: Optional[ProcessingSettings]
    articulated_objects: Optional[ProcessingSettings]

    def __init__(
        self,
        name: str,
        dataset_config: str,
        objects: Optional[ProcessingSettings] = None,
        articulated_objects: Optional[ProcessingSettings] = None,
        stages: Optional[ProcessingSettings] = None,
        scene_whitelist: Optional[list[str]] = None,
        scene_blacklist: Optional[list[str]] = None,
        include_orphan_assets=True,
    ):
        self.name = name
        self.dataset_config = dataset_config
        self.objects = objects
        self.articulated_objects = articulated_objects
        self.stages = stages
        self.scene_whitelist = scene_whitelist
        self.scene_blacklist = scene_blacklist
        self.include_orphan_assets = include_orphan_assets


@dataclass
class AssetSource:
    """
    Defines a data source.
    """

    name: str
    """
    Name of the source.
    """

    assets: list[str]
    """
    List of assets to import. Uses 'glob' format.
    Examples:
    - "data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf"
    - "data/humanoids/humanoid_data/**/*.glb"
    """

    settings: ProcessingSettings


@dataclass
class Config:
    input_dir: str
    """Path of the input `data/` folder to process."""

    data_folder_name: str
    """Name of the input `data/` folder."""

    output_dir: str
    """Path to the output directory."""

    verbose: bool
    """Increase verbosity."""

    use_multiprocessing: bool
    """Use parallelism to speed-up processing. Disable when debugging."""

    datasets: list[HabitatDatasetSource]
    additional_assets: list[AssetSource]
