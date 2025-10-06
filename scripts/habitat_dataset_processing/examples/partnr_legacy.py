#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This script processes graphics assets for WebGL partnr applications with legacy episode sets.

<!> This assumes legacy dataset paths and versions! <!>
* fpss:
    * Source: https://huggingface.co/datasets/fpss/fphab
    * Path: `data/fpss`
    * Branch: siro-benchmark-05-24
* ovmm_objects:
    * Source: https://huggingface.co/datasets/ai-habitat/OVMM_objects
    * Path: `data/objects_ovmm` (beware the reversed name)
"""

from habitat_dataset_processing import (
    AssetPipeline,
    AssetSource,
    GroupType,
    HabitatDatasetSource,
    Operation,
    ProcessingSettings,
)

if __name__ == "__main__":
    object_dataset_settings = ProcessingSettings(
        operation=Operation.PROCESS,
        decimate=False,
        group=GroupType.GROUP_BY_CHUNK,
    )

    datasets: list[HabitatDatasetSource] = [
        # fpss
        HabitatDatasetSource(
            name="fpss",
            dataset_config="fpss/hssd-hab-siro-filtered.scene_dataset_config.json",
            stages=ProcessingSettings(
                operation=Operation.PROCESS,
                decimate=False,
                group=GroupType.GROUP_BY_SCENE,
            ),
            objects=ProcessingSettings(
                operation=Operation.PROCESS,
                decimate=True,
                group=GroupType.GROUP_BY_SCENE,
            ),
            articulated_objects=ProcessingSettings(
                operation=Operation.PROCESS,
                decimate=False,
                group=GroupType.GROUP_BY_SCENE,
            ),
            include_orphan_assets=False,
        ),
        # ai2thorhab
        HabitatDatasetSource(
            name="ai2thor_object_dataset",
            dataset_config="objects_ovmm/train_val/ai2thorhab/ai2thor_object_dataset.scene_dataset_config.json",
            objects=object_dataset_settings,
        ),
        # amazon_berkeley
        HabitatDatasetSource(
            name="amazon_berkeley",
            dataset_config="objects_ovmm/train_val/amazon_berkeley/amazon_berkeley.scene_dataset_config.json",
            objects=object_dataset_settings,
        ),
        # google_scanned
        HabitatDatasetSource(
            name="google_scanned",
            dataset_config="objects_ovmm/train_val/google_scanned/google_scanned.scene_dataset_config.json",
            objects=object_dataset_settings,
        ),
        # hssd-objects
        HabitatDatasetSource(
            name="hssd",
            dataset_config="objects_ovmm/train_val/hssd/hssd.scene_dataset_config.json",
            objects=object_dataset_settings,
        ),
    ]

    additional_assets: list[AssetSource] = [
        # spot
        AssetSource(
            name="hab_spot_arm",
            assets=["robots/hab_spot_arm/urdf/hab_spot_arm.urdf"],
            settings=ProcessingSettings(
                operation=Operation.PROCESS,
                decimate=False,
                group=GroupType.LOCAL,
            ),
        ),
        # humanoids
        AssetSource(
            name="humanoids",
            assets=["humanoids/humanoid_**/*.glb"],
            settings=ProcessingSettings(
                operation=Operation.COPY,
                decimate=False,
                group=GroupType.LOCAL,
            ),
        ),
    ]

    asset_pipeline = AssetPipeline(
        datasets=datasets,
        additional_assets=additional_assets,
        output_subdir="partnr_legacy",
    )
    asset_pipeline.process()
