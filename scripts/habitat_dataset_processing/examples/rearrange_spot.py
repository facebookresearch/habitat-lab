#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This script processes graphics assets for: `hssd-hab`, `ovmm_objects`, `hab_spot_arm` and `humanoids`.
Decimation is disabled.

To procure the datasets, run the following command from your Habitat environment:
```
python -m habitat_sim.utils.datasets_download --uids hssd-hab ovmm_objects hab_spot_arm habitat_humanoids
```

To process the data folder, run:
```
python rearrange_spot.py --input path/to/data
```
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
        # hssd-hab
        HabitatDatasetSource(
            name="hssd-hab-articulated",
            dataset_config="hssd-hab/hssd-hab-articulated.scene_dataset_config.json",
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
            dataset_config="objects/ovmm_objects/train_val/ai2thorhab/ai2thor_object_dataset.scene_dataset_config.json",
            objects=object_dataset_settings,
        ),
        # amazon_berkeley
        HabitatDatasetSource(
            name="amazon_berkeley",
            dataset_config="objects/ovmm_objects/train_val/amazon_berkeley/amazon_berkeley.scene_dataset_config.json",
            objects=object_dataset_settings,
        ),
        # google_scanned
        HabitatDatasetSource(
            name="google_scanned",
            dataset_config="objects/ovmm_objects/train_val/google_scanned/google_scanned.scene_dataset_config.json",
            objects=object_dataset_settings,
        ),
        # hssd-objects
        HabitatDatasetSource(
            name="hssd",
            dataset_config="objects/ovmm_objects/train_val/hssd/hssd.scene_dataset_config.json",
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
        output_subdir="graphics",
    )
    asset_pipeline.process()
