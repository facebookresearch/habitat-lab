#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
        decimate=True,
        group=GroupType.LOCAL,
    )

    datasets: list[HabitatDatasetSource] = [
        # hssd-hab
        HabitatDatasetSource(
            name="hssd-hab-articulated",
            dataset_config="hssd-hab/hssd-hab-articulated.scene_dataset_config.json",
            stages=ProcessingSettings(
                operation=Operation.PROCESS,
                decimate=False,
                group=GroupType.LOCAL,
            ),
            objects=ProcessingSettings(
                operation=Operation.PROCESS,
                decimate=True,
                group=GroupType.LOCAL,
            ),
            articulated_objects=ProcessingSettings(
                operation=Operation.PROCESS,
                decimate=False,
                group=GroupType.LOCAL,
            ),
            scene_whitelist = [
                "104862660_172226844",
                "103997970_171031287",
                "104862669_172226853",
                "102344193",
                "108294897_176710602",
                "104862501_172226556",
                "108736722_177263382",
                "107734254_176000121",
                "108736824_177263559",
                "103997643_171030747",
                "108294558_176710095",
                "108736872_177263607",
                "108294939_176710668",
                "104862621_172226772",
                "106879080_174887211",
                "106366104_174226332",
                "104862681_172226874",
                "103997895_171031182",
                "104348133_171513054",
                "106878945_174887058",
                "108736635_177263256",
                "108736884_177263634",
            ],
            include_orphan_assets=False,
            allow_errors=True,
        ),
        # ycb
        HabitatDatasetSource(
            name="ycb",
            dataset_config="objects/ycb/ycb.scene_dataset_config.json",
            objects=object_dataset_settings,
        ),
        # knuckles
        HabitatDatasetSource(
            name="Fremont-Knuckles",
            dataset_config="Fremont-Knuckles/fremont_knuckles.scene_dataset_config.json",
            stages=ProcessingSettings(
                operation=Operation.PROCESS,
                decimate=False,
                group=GroupType.LOCAL,
            ),
            objects=ProcessingSettings(
                operation=Operation.PROCESS,
                decimate=True,
                group=GroupType.LOCAL,
            ),
            articulated_objects=ProcessingSettings(
                operation=Operation.PROCESS,
                decimate=False,
                group=GroupType.LOCAL,
            ),
            include_orphan_assets=False,
            allow_errors=True,
        )
    ]

    additional_assets: list[AssetSource] = [
        # murp
        AssetSource(
            name="murp_tmr_franka_metahand",
            assets=[
                "hab_murp/murp_tmr_franka/franka_with_hand_v2.3.urdf"
            ],
            settings=ProcessingSettings(
                operation=Operation.PROCESS,
                decimate=False,
                group=GroupType.LOCAL,
            ),
        ),
    ]

    asset_pipeline = AssetPipeline(
        datasets=datasets,
        additional_assets=additional_assets,
        output_subdir="murp_teleop_vr_gfx",
    )
    asset_pipeline.process()
